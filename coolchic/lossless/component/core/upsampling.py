# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md
from __future__ import annotations

from typing import List, OrderedDict

import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from einops import rearrange
from torch import Tensor, nn


class _Parameterization_Symmetric_1d(nn.Module):
    """This module is not meant to be instantiated. It should rather be used
    through the ``torch.nn.utils.parametrize.register_parametrization()``
    function to reparameterize a N-element vector into a 2N-element (or 2N+1)
    symmetric vector. For instance:

        * x = a b c and target_k_size = 5 --> a b c b a
        * x = a b c and target_k_size = 6 --> a b c c b a

    Both these 5-element or 6-element vectors can be parameterize through
    a 3-element representation (a, b, c).
    """

    def __init__(self, target_k_size: int):
        """
        Args:
            target_k_size: Target size of the kernel after reparameterization.
        """

        super().__init__()
        self.target_k_size = target_k_size
        self.param_size = _Parameterization_Symmetric_1d.size_param_from_target(
            self.target_k_size
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return a longer, symmetric vector by concatenating x with a flipped
        version of itself.

        Args:
            x (Tensor): [N] tensor.

        Returns:
            Tensor: [2N] or [2N + 1] tensor, depending on self.target_k_size
        """

        # torch.fliplr requires to have a 2D kernel
        x_reversed = torch.fliplr(x.view(1, -1)).view(-1)

        kernel = torch.cat(
            [
                x,
                # a b c c b a if n is even or a b c b a if n is odd
                x_reversed[self.target_k_size % 2 :],
            ],
        )

        return kernel

    @classmethod
    def size_param_from_target(cls, target_k_size: int) -> int:
        """Return the size of the appropriate parameterization of a
        symmetric tensor with target_k_size elements. For instance:

            target_k_size = 6 ; parameterization size = 3 e.g. (a b c c b a)

            target_k_size = 7 ; parameterization size = 4 e.g. (a b c d c b a)

        Args:
            target_k_size (int): Size of the actual symmetric 1D kernel.

        Returns:
            int: Size of the underlying parameterization.
        """
        # For a kernel of size target_k_size = 2N, we need N values
        # e.g. 3 params a b c to parameterize a b c c b a.
        # For a kernel of size target_k_size = 2N + 1, we need N + 1 values
        # e.g. 4 params a b c d to parameterize a b c d c b a.
        return (target_k_size + 1) // 2


class UpsamplingSeparableSymmetricConv2d(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()

        assert (
            kernel_size % 2 == 1
        ), f"Upsampling kernel size must be odd, found {kernel_size}."

        self.target_k_size = kernel_size
        self.param_size = _Parameterization_Symmetric_1d.size_param_from_target(
            self.target_k_size
        )

        # -------- Instantiate empty parameters, set by the initialize function
        self.weight = nn.Parameter(
            torch.empty(self.param_size), requires_grad=True
        )

        self.bias = nn.Parameter(torch.empty(1), requires_grad=True)
        self.initialize_parameters()
        # -------- Instantiate empty parameters, set by the initialize function

    def initialize_parameters(self) -> None:
        if parametrize.is_parametrized(self, "weight"):
            parametrize.remove_parametrizations(
                self, "weight", leave_parametrized=False
            )

        # Zero everywhere except for the last coef
        w = torch.zeros_like(self.weight)
        w[-1] = 1
        self.weight = nn.Parameter(w, requires_grad=True)

        self.bias = nn.Parameter(
            torch.zeros_like(self.bias), requires_grad=True
        )

        # Each time we call .weight, we'll call the forward of
        # _Parameterization_Symmetric_1d to get a symmetric kernel.
        parametrize.register_parametrization(
            self,
            "weight",
            _Parameterization_Symmetric_1d(target_k_size=self.target_k_size),
            # Unsafe because we change the data dimension, from N to 2N + 1
            unsafe=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        k = self.weight.size()[0]
        weight = self.weight.view(1, -1)
        padding = k // 2

        # If zero channel, there is no data --> nothing to do!
        if x.size()[1] == 0:
            return x

        # Train using non-separable (more stable)
        if self.training:
            # Kronecker product of (1 k) & (k 1) --> (k, k).
            # Then, two dummy dimensions are added to be compliant with conv2d
            # (k, k) --> (1, 1, k, k).
            kernel_2d = torch.kron(weight, weight.T).view((1, 1, k, k))

            # ! Note the residual connection!
            return (
                F.conv2d(x, kernel_2d, bias=None, stride=1, padding=padding) + x
            )

        # Test through separable (less complex, for the flop counter)
        else:
            yw = F.conv2d(x, weight.view((1, 1, 1, k)), padding=(0, padding))

            # ! Note the residual connection!
            return (
                F.conv2d(yw, weight.view((1, 1, k, 1)), padding=(padding, 0))
                + x
            )


class UpsamplingSeparableSymmetricConvTranspose2d(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()

        assert (
            kernel_size >= 4 and not kernel_size % 2
        ), f"Upsampling kernel size shall be even and ≥4. Found {kernel_size}"

        self.target_k_size = kernel_size
        self.param_size = _Parameterization_Symmetric_1d.size_param_from_target(
            self.target_k_size
        )

        # -------- Instantiate empty parameters, set by the initialize function
        self.weight = nn.Parameter(
            torch.empty(self.param_size), requires_grad=True
        )

        self.bias = nn.Parameter(torch.empty(1), requires_grad=True)
        self.initialize_parameters()
        # -------- Instantiate empty parameters, set by the initialize function

    def initialize_parameters(self) -> None:
        if parametrize.is_parametrized(self, "weight"):
            parametrize.remove_parametrizations(
                self, "weight", leave_parametrized=False
            )

        # For a target kernel size of 4 or 6, we use a bilinear kernel as the
        # initialization. For bigger kernels, a bicubic kernel is used. In both
        # case we just initialize the left half of the kernel since these
        # filters are symmetrical
        if self.target_k_size < 8:
            kernel_core = torch.tensor([1.0 / 4.0, 3.0 / 4.0])
        else:
            kernel_core = torch.tensor(
                [0.0351562, 0.1054687, -0.2617187, -0.8789063]
            )

        # If target_k_size = 6, then param_size = 3 while kernel_core = 2
        # Thus we need to add zero_pad = 1 to the left of the kernel.
        zero_pad = self.param_size - kernel_core.size()[0]
        w = torch.zeros_like(self.weight)
        w[zero_pad:] = kernel_core
        self.weight = nn.Parameter(w, requires_grad=True)

        self.bias = nn.Parameter(
            torch.zeros_like(self.bias), requires_grad=True
        )

        # Each time we call .weight, we'll call the forward of
        # _Parameterization_Symmetric_1d to get a symmetric kernel.
        parametrize.register_parametrization(
            self,
            "weight",
            _Parameterization_Symmetric_1d(target_k_size=self.target_k_size),
            # Unsafe because we change the data dimension, from N to 2N + 1
            unsafe=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        k = self.target_k_size  # kernel size
        P0 = k // 2  # could be 0 or k//2 as in legacy implementation
        C = (
            2 * P0 - 1 + k // 2
        )  # crop side border k - 1 + k//2 (k=4, C=5  k=8, C=11)

        weight = self.weight.view(1, -1)

        if self.training:  # training using non-separable (more stable)
            kernel_2d = torch.kron(weight, weight.T).view((1, 1, k, k))

            x_pad = F.pad(x, (P0, P0, P0, P0), mode="replicate")
            yc = F.conv_transpose2d(x_pad, kernel_2d, stride=2)

            # crop to remove padding in convolution
            H, W = yc.size()[-2:]
            y = yc[
                :,
                :,
                C : H - C,
                C : W - C,
            ]

        else:  # testing through separable (less complex)
            # horizontal filtering
            x_pad = F.pad(x, (P0, P0, 0, 0), mode="replicate")
            yc = F.conv_transpose2d(
                x_pad, weight.view((1, 1, 1, k)), stride=(1, 2)
            )
            W = yc.size()[-1]
            y = yc[
                :,
                :,
                :,
                C : W - C,
            ]

            # vertical filtering
            x_pad = F.pad(y, (0, 0, P0, P0), mode="replicate")
            yc = F.conv_transpose2d(
                x_pad, weight.view((1, 1, k, 1)), stride=(2, 1)
            )
            H = yc.size()[-2]
            y = yc[:, :, C : H - C, :]

        return y


class Upsampling(nn.Module):
    def __init__(
        self,
        ups_k_size: int,
        ups_preconcat_k_size: int,
        n_ups_kernel: int,
        n_ups_preconcat_kernel: int,
    ):
        super().__init__()

        # number of kernels for the lower and higher branches
        self.n_ups_kernel = n_ups_kernel
        self.n_ups_preconcat_kernel = n_ups_preconcat_kernel

        # Upsampling kernels = transpose conv2d
        self.conv_transpose2ds = nn.ModuleList(
            [
                UpsamplingSeparableSymmetricConvTranspose2d(ups_k_size)
                for _ in range(n_ups_kernel)
            ]
        )

        # Pre concatenation filters = conv2d
        self.conv2ds = nn.ModuleList(
            [
                UpsamplingSeparableSymmetricConv2d(ups_preconcat_k_size)
                for _ in range(self.n_ups_preconcat_kernel)
            ]
        )

    def forward(self, decoder_side_latent: List[Tensor]) -> Tensor:
        # The main idea is to merge the channel dimension with the batch dimension
        # so that the same convolution is applied independently on the batch dimension.
        latent_reversed = list(reversed(decoder_side_latent))
        upsampled_latent = latent_reversed[0]  # start from smallest

        for idx, target_tensor in enumerate(latent_reversed[1:]):

            # If --n_per_ft_latent = 0,0,1,1,1 return a [1, 3, H/4, W/4] tensor
            if target_tensor.size()[1] == 0:
                break

            # Our goal is to upsample <upsampled_latent> to the same resolution than <target_tensor>
            x = rearrange(upsampled_latent, "b c h w -> (b c) 1 h w")
            x = self.conv_transpose2ds[idx % self.n_ups_kernel](x)

            x = rearrange(
                x, "(b c) 1 h w -> b c h w", b=upsampled_latent.shape[0]
            )
            # Crop to comply with higher resolution feature maps size before concatenation
            x = x[:, :, : target_tensor.shape[-2], : target_tensor.shape[-1]]

            high_branch = self.conv2ds[idx % self.n_ups_preconcat_kernel](
                target_tensor
            )
            upsampled_latent = torch.cat((high_branch, x), dim=1)

        return upsampled_latent

    def get_param(self) -> OrderedDict[str, Tensor]:
        # Detach & clone to create a copy
        return OrderedDict(
            {k: v.detach().clone() for k, v in self.named_parameters()}
        )

    def set_param(self, param: OrderedDict[str, Tensor]):
        self.load_state_dict(param)

    def reinitialize_parameters(self) -> None:
        for i in range(len(self.conv_transpose2ds)):
            self.conv_transpose2ds[i].initialize_parameters()
        for i in range(len(self.conv2ds)):
            self.conv2ds[i].initialize_parameters()
