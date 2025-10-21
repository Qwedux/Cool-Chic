# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


from typing import OrderedDict, Tuple
from lossless.component.core.arm import (
    _get_neighbor,
    _get_non_zero_pixel_ctx_index,
)
import torch

# import torch.nn.functional as F
from torch import Tensor, nn  # , index_select
from lossless.component.core.arm import ArmLinear, _get_neighbor


class ImageArm(nn.Module):
    """Instantiate an autoregressive probability module, modelling the
    conditional distribution :math:`p_{\\psi}(\\hat{y}_i \\mid
    \\mathbf{c}_i)` of a (quantized) latent pixel :math:`\\hat{y}_i`,
    conditioned on neighboring already decoded context pixels
    :math:`\\mathbf{c}_i \in \\mathbb{Z}^C`, where :math:`C` denotes the
    number of context pixels.

    The distribution :math:`p_{\\psi}` is assumed to follow a Laplace
    distribution, parameterized by an expectation :math:`\\mu` and a scale
    :math:`b`, where the scale and the variance :math:`\\sigma^2` are
    related as follows :math:`\\sigma^2 = 2 b ^2`.

    The parameters of the Laplace distribution for a given latent pixel
    :math:`\\hat{y}_i` are obtained by passing its context pixels
    :math:`\\mathbf{c}_i` through an MLP :math:`f_{\\psi}`:

    .. math::

        p_{\\psi}(\\hat{y}_i \\mid \\mathbf{c}_i) \sim \mathcal{L}(\\mu_i,
        b_i), \\text{ where } \\mu_i, b_i = f_{\\psi}(\\mathbf{c}_i).

    .. attention::

        The MLP :math:`f_{\\psi}` has a few constraint on its architecture:

        * The width of all hidden layers (i.e. the output of all layers except
          the final one) are identical to the number of pixel contexts
          :math:`C`;

        * All layers except the last one are residual layers, followed by a
          ``ReLU`` non-linearity;

        * :math:`C` must be at a multiple of 8.

    The MLP :math:`f_{\\psi}` is made of custom Linear layers instantiated
    from the ``ArmLinear`` class.
    """

    def __init__(
        self,
        context_size: int = 8,
        n_hidden_layers: int = 2,
        hidden_layer_dim: int = 6,
        synthesis_out_params_per_channel: list[int] = [2, 3, 4],
        channel_separation: bool = True,
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
        assert context_size % 8 == 0, (
            f"ARM context size and hidden layer dimension must be "
            f"a multiple of 8. Found {context_size}."
        )
        self.context_size = context_size
        self.synthesis_out_params_per_channel = synthesis_out_params_per_channel
        self.channel_separation = channel_separation

        if not channel_separation:
            raise NotImplementedError(
                "Non channel-separated ARM is not implemented yet."
            )

        # ======================== Construct the MLPs ======================== #
        self.model_layers = [
            nn.ModuleList()
            for _ in range(len(self.synthesis_out_params_per_channel))
        ]
        self.models = nn.ModuleList(
            nn.Sequential()
            for _ in range(len(self.synthesis_out_params_per_channel))
        )
        for channel_idx, output_dim in enumerate(
            self.synthesis_out_params_per_channel
        ):
            self.model_layers[channel_idx].append(
                ArmLinear(
                    context_size
                    * len(
                        self.synthesis_out_params_per_channel
                    )  # context size * num_channels
                    + sum(
                        self.synthesis_out_params_per_channel
                    )  # we can use all information from synthesis output
                    + channel_idx,  # extra information from already decoded channels for current pixel
                    hidden_layer_dim,
                    residual=False,
                )
            )
            self.model_layers[channel_idx].append(nn.ReLU())

            # Construct the hidden layer(s)
            for _ in range(n_hidden_layers - 1):
                self.model_layers[channel_idx].append(
                    ArmLinear(hidden_layer_dim, hidden_layer_dim, residual=True)
                )
                self.model_layers[channel_idx].append(nn.ReLU())
            # Construct the output layer. It always has output_dim 2*outputs
            # since we use the second half for gating
            self.model_layers[channel_idx].append(
                ArmLinear(hidden_layer_dim, output_dim * 2, residual=False)
            )
            self.models[channel_idx] = nn.Sequential(
                *self.model_layers[channel_idx]
            )

        self.mask_size = 9
        self.register_buffer(
            "non_zero_image_arm_ctx_index",
            _get_non_zero_pixel_ctx_index(self.context_size),
            persistent=False,
        )

    def prepare_inputs(self, image: Tensor, raw_synth_out: Tensor):
        contexts = []
        assert len(self.synthesis_out_params_per_channel) == image.shape[1], (
            "Number of channels in image and synthesis_out_params_per_channel "
            "must be equal."
        )

        # First get contexts for all channels in the image
        # Use loop as _get_neighbor supports only [1, 1, H, W] input shape
        for channel_idx in range(len(self.synthesis_out_params_per_channel)):
            contexts.append(
                _get_neighbor(
                    image[:, channel_idx : channel_idx + 1, :, :],
                    self.mask_size,
                    self.non_zero_image_arm_ctx_index,  # type: ignore
                )
            )
        # Now concatenate the num_channels [H *W, context_size] shaped image contexts
        # into [H *W, context_size * num_channels]
        flat_image_context = torch.cat(contexts, dim=1)

        # Add synthesis output and already decoded channels information
        prepared_inputs = []
        for channel_idx in range(len(self.synthesis_out_params_per_channel)):
            prepared_inputs.append(
                torch.cat(
                    [
                        flat_image_context,
                        # synthesis output has shape [1, C, H, W], we want [H*W, C]
                        raw_synth_out.permute(0, 2, 3, 1).reshape(
                            -1, sum(self.synthesis_out_params_per_channel)
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
                                requires_grad=True,
                            )
                        ),
                    ],
                    dim=1,
                )
            )
        return prepared_inputs

    def forward(self, x: Tensor, synthesis_proba: Tensor) -> Tensor:
        """Perform the auto-regressive module (ARM) forward pass. The ARM takes
        as input a tensor of shape :math:`[B, C]` i.e. :math:`B` contexts with
        :math:`C` context pixels. ARM outputs :math:`[B, 2]` values correspond
        to :math:`\\mu, b` for each of the :math:`B` input pixels.

        .. warning::

            Note that the ARM expects input to be flattened i.e. spatial
            dimensions :math:`H, W` are collapsed into a single batch-like
            dimension :math:`B = HW`, leading to an input of shape
            :math:`[B, C]`, gathering the :math:`C` contexts for each of the
            :math:`B` pixels to model.

        .. note::

            The ARM MLP does not output directly the scale :math:`b`. Denoting
            :math:`s` the raw output of the MLP, the scale is obtained as
            follows:

            .. math::

                b = e^{x - 4}

        Args:
            x: Concatenation of all input contexts
                :math:`\\mathbf{c}_i`. Tensor of shape :math:`[B, C]`.

        Returns:
            Concatenation of all Laplace distributions param :math:`\\mu, b`.
            Tensor of shape :math:([B]). Also return the *log scale*
            :math:`s` as described above. Tensor of shape :math:`(B)`
        """
        prepared_inputs = self.prepare_inputs(x, synthesis_proba)
        cutoffs = [
            sum(self.synthesis_out_params_per_channel[:i])
            for i in range(len(self.synthesis_out_params_per_channel) + 1)
        ]
        out_probas_param = []
        for channel in range(len(self.synthesis_out_params_per_channel)):
            raw_outs = self.models[channel](prepared_inputs[channel])
            raw_proba_param, gate = raw_outs.chunk(2, dim=1)
            out_probas_param.append(
                synthesis_proba.permute(0, 2, 3, 1).reshape(
                    -1, sum(self.synthesis_out_params_per_channel)
                )[:, cutoffs[channel] : cutoffs[channel + 1]]
                + raw_proba_param * torch.sigmoid(gate)
            )
        out_proba_param = torch.cat(out_probas_param, dim=1)
        reshaped_image_arm_out = out_proba_param.view(
            synthesis_proba.shape[0],
            synthesis_proba.shape[2],
            synthesis_proba.shape[3],
            synthesis_proba.shape[1],
        ).permute(0, 3, 1, 2)

        return reshaped_image_arm_out

    # def get_param(self) -> OrderedDict[str, Tensor]:
    #     """Return **a copy** of the weights and biases inside the module.

    #     Returns:
    #         A copy of all weights & biases in the layers.
    #     """
    #     # Detach & clone to create a copy
    #     return OrderedDict(
    #         {k: v.detach().clone() for k, v in self.named_parameters()}
    #     )

    def set_param(self, param: OrderedDict[str, Tensor]) -> None:
        """Replace the current parameters of the module with param.

        Args:
            param: Parameters to be set.
        """
        self.load_state_dict(param)

    def reinitialize_parameters(self) -> None:
        """Re-initialize in place the parameters of all the ArmLinear layer."""
        for layer in self.mlp.children():
            if isinstance(layer, ArmLinear):
                layer.initialize_parameters()
