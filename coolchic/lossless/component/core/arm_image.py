# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md
from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import OrderedDict, assert_never

import numpy as np
import torch
from lossless.component.core.arm import (ArmLinear, _get_neighbor,
                                         _get_non_zero_pixel_ctx_index)
from lossless.component.types import (IMARM_SPLIT_DIRECTION, HorizontalSplit,
                                      VerticalSplit)
from lossless.util.image import ImageHeight, ImageWidth
from lossless.util.misc import safe_get_from_nested_lists
from torch import Tensor, nn


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


@dataclass(frozen=True, kw_only=True)
class MultiImageArmDescriptor:
    """For now I assume a simple case where the image is split into equally sized
    cells in a grid like fashion."""

    # value at index k says which pixels are assigned to expert k
    expert_indices: Sequence[Tensor]
    image_height: torch.Tensor
    image_width: torch.Tensor
    # value of cell (i,j) says which expert to use for that pixel position
    routing_grid: torch.Tensor
    presets: dict
    active_preset: str

    @cached_property
    def num_experts(self) -> int:
        return torch.unique(self.routing_grid).numel()

    @staticmethod
    def compute_expert_indices(routing_grid: Tensor) -> Sequence[Tensor]:
        router_flat = routing_grid.flatten()
        num_experts = torch.unique(routing_grid).numel()
        return [
            torch.where(router_flat == i)[0] for i in range(num_experts)
        ]

    @staticmethod
    def simple_grid_routing(image_height: ImageHeight, image_width: ImageWidth, num_parts_per_col: int, num_parts_per_row: int) -> Tensor:
        """
        Makes a simple grid that looks like this for 2x2 parts:
        [[0, 0, 1, 1],
         [0, 0, 1, 1],
         [2, 2, 3, 3],
         [2, 2, 3, 3]]
        """
        tmp_routing_grid = np.zeros((image_height, image_width), dtype=int)

        # Get the row and column groupings
        # Single grouping can look like this: [0,0,0,1,1] for 2 parts over 5 pixels
        # The final routing grid is then product of row and column groupings
        row_groups_indices, _ = groups_and_sizes_from_num_splits(
            num_parts_per_row, int(image_width)
        )
        row_group_mapping = {i: row_groups_indices[i] for i in range(len(row_groups_indices))}
        col_groups_indices, _ = groups_and_sizes_from_num_splits(
            num_parts_per_col, int(image_height)
        )
        col_group_mapping = {i: col_groups_indices[i] for i in range(len(col_groups_indices))}
        for i in range(int(image_height)):
            for j in range(int(image_width)):
                tmp_routing_grid[i, j] = (
                    col_group_mapping[i] * num_parts_per_row + row_group_mapping[j]
                )
        return torch.tensor(tmp_routing_grid, dtype=torch.int)

    @staticmethod
    def split_expert_region(
        multi_image_arm_descriptor: MultiImageArmDescriptor, expert_idx: int, split_direction: IMARM_SPLIT_DIRECTION
    ) -> MultiImageArmDescriptor:
        expert_region_mask = (
            multi_image_arm_descriptor.routing_grid == expert_idx
        )
        ys, xs = torch.where(expert_region_mask)
        min_y, max_y = ys.min().item(), ys.max().item()
        min_x, max_x = xs.min().item(), xs.max().item()
        height = max_y - min_y + 1
        width = max_x - min_x + 1

        new_routing_grid = multi_image_arm_descriptor.routing_grid.clone()
        new_expert_idx = torch.unique(new_routing_grid).numel()

        match split_direction:
            case HorizontalSplit():
                x_start = min_x + (width // 2 + width % 2)
                y_start = min_y
            case VerticalSplit():
                x_start = min_x
                y_start = min_y + (height // 2 + height % 2)
            case _:
                assert_never(split_direction)

        new_routing_grid[y_start:max_y + 1, x_start:max_x + 1][
            expert_region_mask[y_start:max_y + 1, x_start:max_x + 1]
        ] = (
            new_expert_idx
        )
        num_experts = torch.unique(new_routing_grid).numel()
        expert_indices = MultiImageArmDescriptor.compute_expert_indices(new_routing_grid)
        return MultiImageArmDescriptor(
            expert_indices=expert_indices,
            image_height=multi_image_arm_descriptor.image_height,
            image_width=multi_image_arm_descriptor.image_width,
            routing_grid=new_routing_grid,
            presets=multi_image_arm_descriptor.presets,
            active_preset=multi_image_arm_descriptor.active_preset
        )


@dataclass(frozen=True, kw_only=True)
class ImageARMParameter:
    context_size: int
    n_hidden_layers: int
    hidden_layer_dim: int
    synthesis_out_params_per_channel: Sequence[int]
    use_color_regression: bool
    multi_region_image_arm_specification: MultiImageArmDescriptor | None




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
        """
        super().__init__()
        self.params = params
        assert self.params.multi_region_image_arm_specification is not None, "Multi-region image ARM specification is required"
        assert self.params.context_size % 8 == 0, (
            f"ARM context size and hidden layer dimension must be "
            f"a multiple of 8. Found {self.params.context_size}."
        )
        # ======================== Construct the MLPs ======================== #
        self.image_arm_models: nn.ModuleList = nn.ModuleList(
            nn.ModuleList(
                self._create_image_arm_network(channel_idx, channel_output_dim)
                for channel_idx, channel_output_dim in enumerate(
                    self.params.synthesis_out_params_per_channel
                )
            )
            for _ in range(self.params.multi_region_image_arm_specification.num_experts)
        )

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

    def reinitialize_image_arm_experts(self, num_experts: int, pretrained_expert_index: int) -> None:
        """Re-initialize the image ARM experts.
        """
        with torch.no_grad():
            pretrained_expert = copy.deepcopy(self.image_arm_models[pretrained_expert_index])
            self.image_arm_models = nn.ModuleList(
                copy.deepcopy(pretrained_expert) for _ in range(num_experts)
            )

    def split_image_arm_expert(
        self, expert_idx: int, split_direction: IMARM_SPLIT_DIRECTION
    ) -> None:
        expert_region_model = self.image_arm_models[expert_idx]
        assert isinstance(expert_region_model, nn.ModuleList)
        # Duplicate the model for the new experts
        self.image_arm_models.append(
            copy.deepcopy(expert_region_model)
        )
        # Update the routing grid
        self.params.multi_region_image_arm_specification.split_expert_region(
            expert_idx, split_direction
        )

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
        raw_synth_out_flat = raw_synth_out.permute(0, 2, 3, 1).reshape(
            -1, sum(self.params.synthesis_out_params_per_channel)
        )
        out_probas_param_flat = torch.zeros_like(
            raw_synth_out_flat, dtype=raw_synth_out.dtype, device=raw_synth_out.device
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
                gated_value = raw_proba_param * torch.sigmoid(gate)

                # Update the flattened output tensor
                out_probas_param_flat[indices, cutoffs[channel] : cutoffs[channel + 1]] = (
                    raw_synth_out_flat[indices, cutoffs[channel] : cutoffs[channel + 1]]
                    + gated_value
                )

        # Reshape back to [1, H, W, S] then permute to [1, S, H, W]
        reshaped_image_arm_out = out_probas_param_flat.view(
            raw_synth_out.shape[0], raw_synth_out.shape[2], raw_synth_out.shape[3], -1
        ).permute(0, 3, 1, 2)

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
        # Keep context on the same device as the source tensor (CPU/GPU).
        if isinstance(grid_so_far, torch.Tensor):
            return torch.tensor(
                neighbor_context,
                dtype=grid_so_far.dtype,
                device=grid_so_far.device,
            ).unsqueeze(0)
        # Fallback for list-based callers.
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
