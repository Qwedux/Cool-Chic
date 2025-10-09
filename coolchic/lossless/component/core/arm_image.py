# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


from typing import OrderedDict, Tuple

import torch

# import torch.nn.functional as F
from torch import Tensor, nn  # , index_select
from lossless.component.core.arm import ArmLinear


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

    def __init__(self, dim_arm: int, output_dim: int, n_hidden_layers_arm: int):
        """
        Args:
            dim_arm: Number of context pixels AND dimension of all hidden
                layers :math:`C`.
            n_hidden_layers_arm: Number of hidden layers. Set it to 0 for
                a linear ARM.
        """
        super().__init__()
        assert dim_arm % 8 == 0, (
            f"ARM context size and hidden layer dimension must be "
            f"a multiple of 8. Found {dim_arm}."
        )
        self.dim_arm = dim_arm

        # ======================== Construct the MLP ======================== #
        layers_list = nn.ModuleList()
        # we have dim_arm context pixels and output_dim parameters to residualy correct
        layers_list.append(
            ArmLinear(dim_arm + output_dim, dim_arm, residual=False)
        )
        layers_list.append(nn.ReLU())

        # Construct the hidden layer(s)
        for i in range(n_hidden_layers_arm - 1):
            layers_list.append(ArmLinear(dim_arm, dim_arm, residual=True))
            layers_list.append(nn.ReLU())

        # Construct the output layer. It always has 2 outputs (mu and scale)
        layers_list.append(ArmLinear(dim_arm, output_dim * 2, residual=False))
        self.mlp = nn.Sequential(*layers_list)
        # ======================== Construct the MLP ======================== #

    def forward(
        self, x: Tensor, synthesis_proba: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
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
        raw_out = self.mlp(torch.cat([x, synthesis_proba], dim=1))
        raw_proba_param, gate = raw_out.chunk(2, dim=1)
        out_proba_param = synthesis_proba + raw_proba_param * torch.sigmoid(
            gate
        )

        return out_proba_param

    def get_param(self) -> OrderedDict[str, Tensor]:
        """Return **a copy** of the weights and biases inside the module.

        Returns:
            A copy of all weights & biases in the layers.
        """
        # Detach & clone to create a copy
        return OrderedDict(
            {k: v.detach().clone() for k, v in self.named_parameters()}
        )

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
