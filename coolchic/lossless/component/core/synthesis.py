# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import OrderedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SynthesisConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int,
        residual: bool = False,
    ):
        super().__init__()

        self.residual = residual
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pad = int((kernel_size - 1) / 2)

        self.groups = groups
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // self.groups, kernel_size, kernel_size),
            requires_grad=True,
        )
        self.bias = nn.Parameter(torch.empty((out_channels)), requires_grad=True)
        self.initialize_parameters()

    def forward(self, x: Tensor) -> Tensor:
        padded_x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="replicate")
        y = F.conv2d(padded_x, self.weight, self.bias, groups=self.groups)

        if self.residual:
            return y + x
        else:
            return y

    def initialize_parameters(self) -> None:
        self.bias = nn.Parameter(torch.zeros_like(self.bias), requires_grad=True)

        if self.residual:
            self.weight = nn.Parameter(torch.zeros_like(self.weight), requires_grad=True)
        else:
            out_channel, in_channel_divided_by_group, kernel_height, kernel_weight = (
                self.weight.size()
            )
            in_channel = in_channel_divided_by_group * self.groups
            k = self.groups / (in_channel * kernel_height * kernel_weight)
            sqrt_k = math.sqrt(k)

            self.weight = nn.Parameter(
                (torch.rand_like(self.weight) - 0.5) * 2 * sqrt_k / (out_channel**2),
                requires_grad=True,
            )


class Synthesis(nn.Module):
    possible_non_linearity = {
        "none": nn.Identity,
        "relu": nn.ReLU,
        # "leakyrelu": nn.LeakyReLU,    # Unsupported by the decoder
        # "gelu": nn.GELU,              # Unsupported by the decoder
    }

    possible_mode = ["linear", "residual"]

    def __init__(self, input_ft: int, layers_dim: Sequence[str]):
        super().__init__()

        self.synth_branches = nn.ModuleList()
        self.input_ft = input_ft
        layers_list = nn.ModuleList()

        # Construct the hidden layer(s)
        for layers in layers_dim:
            out_ft, k_size, groups, mode, non_linearity = layers.split("-")
            out_ft = int(out_ft)
            k_size = int(k_size)
            groups = int(groups)

            # Check that mode and non linearity is correct
            assert (
                mode in Synthesis.possible_mode
            ), f"Unknown mode. Found {mode}. Should be in {Synthesis.possible_mode}"

            assert non_linearity in Synthesis.possible_non_linearity, (
                f"Unknown non linearity. Found {non_linearity}. "
                f"Should be in {Synthesis.possible_non_linearity.keys()}"
            )

            # Instantiate them
            layers_list.append(
                SynthesisConv2d(
                    input_ft, out_ft, k_size, groups=groups, residual=mode == "residual"
                )
            )
            layers_list.append(Synthesis.possible_non_linearity[non_linearity]())

            input_ft = out_ft

        self.layers = nn.Sequential(*layers_list)

    def forward(self, x: Tensor) -> Tensor:
        raw_out = self.layers(x)
        return raw_out

    def get_param(self) -> OrderedDict[str, Tensor]:
        return OrderedDict({k: v.detach().clone() for k, v in self.named_parameters()})

    def set_param(self, param: OrderedDict[str, Tensor]):
        self.load_state_dict(param)

    def reinitialize_parameters(self) -> None:
        for layer in self.layers.children():
            if isinstance(layer, SynthesisConv2d):
                layer.initialize_parameters()
