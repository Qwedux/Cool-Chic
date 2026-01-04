# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import itertools
from dataclasses import dataclass, field
from typing import OrderedDict, Tuple

import numpy as np
import torch
from lossless.component.core.arm import (ArmLinear, _get_neighbor,
                                         _get_non_zero_pixel_ctx_index)
from lossless.util.misc import safe_get_from_nested_lists
# import torch.nn.functional as F
from torch import Tensor, nn  # , index_select


def groups_and_sizes_from_num_splits(
    num_splits: int, total_size: int
) -> tuple[list[int], list[int]]:
    base_size = total_size // num_splits
    sizes = [base_size] * num_splits
    remainder = total_size % num_splits
    for i in range(remainder):
        sizes[i] += 1
    group_indices = []
    for i in range(num_splits):
        for _ in range(sizes[i]):
            group_indices.append(i)
    return group_indices, sizes


@dataclass
class MultiImageArmDescriptor:
    """For now I assume a simple case where the image is split into equally sized
    cells in a grid like fashion."""

    expert_indices: list[torch.Tensor] = field(init=False)
    num_arms: torch.Tensor = field(default_factory=lambda: torch.tensor(1))
    image_height: torch.Tensor = field(default_factory=lambda: torch.tensor(0))
    image_width: torch.Tensor = field(default_factory=lambda: torch.tensor(0))
    # value of cell (i,j) says which ARM to use for that pixel position
    routing_grid: torch.Tensor = field(default_factory=lambda: torch.tensor([[0]]))

    def set_image_size(self, image_size: Tuple[int, int]) -> None:
        self.image_height = torch.tensor(image_size[0])
        self.image_width = torch.tensor(image_size[1])

    def simple_grid_routing(self, num_parts_per_col: int, num_parts_per_row: int) -> None:
        self.num_arms = torch.tensor(num_parts_per_row * num_parts_per_col)
        tmp_routing_grid = np.zeros((self.image_height, self.image_width), dtype=int)

        row_groups_indices, _ = groups_and_sizes_from_num_splits(
            num_parts_per_row, int(self.image_width.item())
        )
        row_group_mapping = {i: row_groups_indices[i] for i in range(len(row_groups_indices))}
        col_groups_indices, _ = groups_and_sizes_from_num_splits(
            num_parts_per_col, int(self.image_height.item())
        )
        col_group_mapping = {i: col_groups_indices[i] for i in range(len(col_groups_indices))}
        for i in range(int(self.image_height.item())):
            for j in range(int(self.image_width.item())):
                tmp_routing_grid[i, j] = (
                    col_group_mapping[i] * num_parts_per_row + row_group_mapping[j]
                )
        self.routing_grid = torch.tensor(tmp_routing_grid, dtype=torch.int)
        router_flat = self.routing_grid.flatten()
        self.expert_indices = [
            torch.where(router_flat == i)[0] for i in range(int(self.num_arms.item()))
        ]


@dataclass
class ImageARMParameter:
    context_size: int = 8
    n_hidden_layers: int = 2
    hidden_layer_dim: int = 6
    synthesis_out_params_per_channel: list[int] = field(default_factory=lambda: [2, 2, 2])
    channel_separation: bool = True
    use_color_regression: bool = False
    multi_region_image_arm: bool = False
    multi_region_image_arm_specification: MultiImageArmDescriptor = field(
        default_factory=lambda: MultiImageArmDescriptor()
    )


class ImageArm(nn.Module):
    non_zero_image_arm_ctx_index: torch.Tensor

    def __init__(
        self,
        params: ImageARMParameter,
    ):
        """
        Args:
            context_size: Number of pixels to take into context
            n_hidden_layers: Number of hidden layers. Set it to 0 for
                a linear ARM.
            hidden_layer_dim: Size of hidden layer output
            synthesis_out_params_per_channel: How many values from
                synthesis_out does each channel consume and produce
                (residual=True)
            channel_separation: Use separate networks for each channel. Also use
                information from previous channels appended to the context.
        """
        super().__init__()
        self.params = params
        assert self.params.context_size % 8 == 0, (
            f"ARM context size and hidden layer dimension must be "
            f"a multiple of 8. Found {self.params.context_size}."
        )
        if self.params.use_color_regression:
            self.params.synthesis_out_params_per_channel = [2, 3, 4]

        # ======================== Construct the MLPs ======================== #
        self.image_arm_models = nn.ModuleList(
            nn.ModuleList(
                self._create_image_arm_network(channel_idx, channel_output_dim)
                for channel_idx, channel_output_dim in enumerate(
                    self.params.synthesis_out_params_per_channel
                )
            )
            for _ in range(int(self.params.multi_region_image_arm_specification.num_arms.item()))
        )
        # self.layers = [
        #     nn.ModuleList() for _ in range(len(self.params.synthesis_out_params_per_channel))
        # ]
        # self.models = nn.ModuleList(
        #     nn.Sequential() for _ in range(len(self.params.synthesis_out_params_per_channel))
        # )
        # for channel_idx, output_dim in enumerate(self.params.synthesis_out_params_per_channel):
        #     self.layers[channel_idx].append(
        #         ArmLinear(
        #             self.params.context_size
        #             * len(
        #                 self.params.synthesis_out_params_per_channel
        #             )  # context size * num_channels
        #             + sum(
        #                 self.params.synthesis_out_params_per_channel
        #             )  # we can use all information from synthesis output
        #             + channel_idx,  # extra information from already decoded channels for current pixel
        #             self.params.hidden_layer_dim,
        #             residual=False,
        #         )
        #     )
        #     self.layers[channel_idx].append(nn.ReLU())

        #     # Construct the hidden layer(s)
        #     for _ in range(self.params.n_hidden_layers - 1):
        #         self.layers[channel_idx].append(
        #             ArmLinear(
        #                 self.params.hidden_layer_dim, self.params.hidden_layer_dim, residual=True
        #             )
        #         )
        #         self.layers[channel_idx].append(nn.ReLU())
        #     # Construct the output layer. It always has output_dim 2*outputs
        #     # since we use the second half for gating
        #     self.layers[channel_idx].append(
        #         ArmLinear(self.params.hidden_layer_dim, output_dim * 2, residual=False)
        #     )
        #     self.models[channel_idx] = nn.Sequential(*self.layers[channel_idx])

        self.mask_size = 9
        # we have to register the non zero context indices as buffer for .to(device) calls
        self.register_buffer(
            "non_zero_image_arm_ctx_index",
            _get_non_zero_pixel_ctx_index(self.params.context_size),
            persistent=False,
        )
        self.non_zero_image_arm_ctx_shifts = {
            # row = index // 9, col = index % 9
            index: [index // 9 - 4, index % 9 - 4]
            for index in self.non_zero_image_arm_ctx_index.tolist()
        }

    def _create_image_arm_network(self, channel_idx: int, output_dim: int) -> nn.Sequential:
        layers = []
        layers.append(
            ArmLinear(
                self.params.context_size
                * len(self.params.synthesis_out_params_per_channel)  # context size * num_channels
                + sum(
                    self.params.synthesis_out_params_per_channel
                )  # we can use all information from synthesis output
                + channel_idx,  # extra information from already decoded channels for current pixel
                self.params.hidden_layer_dim,
                residual=False,
            )
        )
        layers.append(nn.ReLU())

        # Construct the hidden layer(s)
        for _ in range(self.params.n_hidden_layers - 1):
            layers.append(
                ArmLinear(self.params.hidden_layer_dim, self.params.hidden_layer_dim, residual=True)
            )
            layers.append(nn.ReLU())
        # Construct the output layer. It always has output_dim 2*outputs
        # since we use the second half for gating
        layers.append(ArmLinear(self.params.hidden_layer_dim, output_dim * 2, residual=False))
        return nn.Sequential(*layers)

    def prepare_inputs(self, image: Tensor, raw_synth_out: Tensor):
        assert len(self.params.synthesis_out_params_per_channel) == image.shape[1], (
            "Number of channels in image and synthesis_out_params_per_channel " "must be equal."
        )

        contexts = []
        # First get contexts for all channels in the image
        # Use loop as _get_neighbor supports only [1, 1, H, W] input shape
        for channel_idx in range(len(self.params.synthesis_out_params_per_channel)):
            contexts.append(
                _get_neighbor(
                    image[:, channel_idx : channel_idx + 1, :, :],
                    self.mask_size,
                    self.non_zero_image_arm_ctx_index,  # type: ignore
                )
            )
        # Now concatenate the num_channels [H *W, context_size] shaped image contexts
        # into [H *W, context_size * num_channels]
        flat_image_context = torch.stack(contexts, dim=2).reshape(
            (image.shape[2] * image.shape[3], -1)
        )

        # Add synthesis output and already decoded channels information
        prepared_inputs = []
        for channel_idx in range(len(self.params.synthesis_out_params_per_channel)):
            prepared_inputs.append(
                torch.cat(
                    [
                        flat_image_context,
                        # synthesis output has shape [1, C, H, W], we want [H*W, C]
                        raw_synth_out.permute(0, 2, 3, 1).reshape(
                            -1, sum(self.params.synthesis_out_params_per_channel)
                        ),
                        # append the couple of already decoded channels
                        (
                            image[:, :channel_idx]
                            .permute(0, 2, 3, 1)
                            .reshape(
                                -1,
                                channel_idx,
                            )
                            if channel_idx > 0
                            else torch.empty(
                                image.shape[2] * image.shape[3],
                                0,
                                dtype=image.dtype,
                                device=image.device,
                                requires_grad=False,
                            )
                        ),
                    ],
                    dim=1,
                )
            )
        return prepared_inputs

    def forward(self, image: Tensor, raw_synth_out: Tensor) -> Tensor:
        """
        Args:
            image: The image tensor of shape [1, C, H, W]
            raw_synth_out: The synthesis output tensor of shape [1, S, H, W] where S is
                sum of synthesis_out_params_per_channel
        Returns:
            The predicted parameters tensor of shape [1, S, H, W]
        """
        # prepared_inputs is a list of length num_channels
        # where i-th element has shape [H*W, image_arm_input_size_for_channel_i]
        prepared_inputs = self.prepare_inputs(image, raw_synth_out)

        cutoffs = [
            sum(self.params.synthesis_out_params_per_channel[:i])
            for i in range(len(self.params.synthesis_out_params_per_channel) + 1)
        ]
        out_probas_param = torch.zeros_like(
            raw_synth_out, dtype=raw_synth_out.dtype, device=raw_synth_out.device
        )
        for expert_idx, indices in enumerate(
            self.params.multi_region_image_arm_specification.expert_indices
        ):
            if len(indices) == 0:
                continue

            for channel in range(len(self.params.synthesis_out_params_per_channel)):
                expert_input = prepared_inputs[channel][indices, :]
                raw_outs = self.image_arm_models[expert_idx][channel](expert_input)  # type: ignore
                raw_proba_param, gate = raw_outs.chunk(2, dim=1)
                # f(inpt) = inpt + res_correction * gate
                out_probas_param[:, cutoffs[channel] : cutoffs[channel + 1], :, :].view(
                    -1, sum(self.params.synthesis_out_params_per_channel)
                )[indices, :] = raw_synth_out.permute(0, 2, 3, 1).reshape(
                    -1, sum(self.params.synthesis_out_params_per_channel)
                )[
                    indices, cutoffs[channel] : cutoffs[channel + 1]
                ] + raw_proba_param * torch.sigmoid(
                    gate
                )

        reshaped_image_arm_out = out_probas_param.permute(0, 3, 1, 2)

        return reshaped_image_arm_out

    def inference_at_position(
        self,
        h: int,
        w: int,
        features: torch.Tensor,
        relevant_raw_synth_out: torch.Tensor,
        channel_idx: int,
    ) -> torch.Tensor:
        expert_idx = int(self.params.multi_region_image_arm_specification.routing_grid[h][w].item())
        raw_outs = self.image_arm_models[expert_idx][channel_idx](features)  # type: ignore
        raw_proba_param, gate = raw_outs.chunk(2, dim=1)
        return relevant_raw_synth_out + raw_proba_param * torch.sigmoid(gate)

    def get_neighbor_context(
        self, grid_so_far: list[list] | torch.Tensor, h: int, w: int
    ) -> Tensor:
        """Get the neighbor context for a given spatial position (h, w)
        in the image grid_so_far.

        Args:
            grid_so_far: The image grid decoded so far of shape [1, C, H, W]
            h: The height position
            w: The width position
        """
        neighbor_context = []
        for idx in self.non_zero_image_arm_ctx_index:
            shift = self.non_zero_image_arm_ctx_shifts[idx.item()]
            neighbor_h = h + shift[0]
            neighbor_w = w + shift[1]
            pixel_value = safe_get_from_nested_lists(
                grid_so_far,
                [neighbor_h, neighbor_w],
                default=0,
            )
            neighbor_context.append(pixel_value)
        # Return as a tensor of shape [1, context_size]
        return torch.tensor(neighbor_context, dtype=torch.float32).unsqueeze(0)

    def get_param(self) -> OrderedDict[str, Tensor]:
        """Get the parameters of the module.

        Returns:
            The parameters of the module.
        """
        return OrderedDict({k: v.detach().clone() for k, v in self.named_parameters()})

    def set_param(self, param: OrderedDict[str, Tensor]) -> None:
        """Replace the current parameters of the module with param.

        Args:
            param: Parameters to be set.
        """
        self.load_state_dict(param)

    def reinitialize_parameters(self) -> None:
        """Re-initialize in place the parameters of all the ArmLinear layers."""
        for module in self.modules():
            if isinstance(module, ArmLinear):
                module.initialize_parameters()
