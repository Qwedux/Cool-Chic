# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

from __future__ import annotations

from typing import Literal, cast

import torch
from torch import Tensor


def softround(x: Tensor, t: Tensor) -> Tensor:
    """
    <https://arxiv.org/pdf/2006.09952.pdf>
    """
    floor_x = torch.floor(x)
    delta = x - floor_x - 0.5
    return floor_x + 0.5 * torch.tanh(delta / t) / torch.tanh(1 / (2 * t)) + 0.5


def generate_kumaraswamy_noise(
    uniform_noise: Tensor, kumaraswamy_param: Tensor
) -> Tensor:
    """
    <https://arxiv.org/abs/2312.02753>
    """
    # This relation between a and b allows to always have a mode of 0.5
    a = kumaraswamy_param
    b = (2**a * (a - 1) + 1) / a

    # Use the inverse of the repartition function to sample a kumaraswamy noise in [0., 1.]
    # Shift the noise to have it in [-0.5, 0.5]
    kumaraswamy_noise = (1 - (1 - uniform_noise) ** (1 / b)) ** (1 / a) - 0.5

    return kumaraswamy_noise


POSSIBLE_QUANTIZATION_NOISE_TYPE = Literal["kumaraswamy", "gaussian", "none"]
POSSIBLE_QUANTIZER_TYPE = Literal["softround_alone", "softround", "hardround", "ste", "none"]


def quantize(
    x: Tensor,
    quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "kumaraswamy",
    quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
    soft_round_temperature: Tensor | None = None,
    noise_parameter: Tensor | None = None,
) -> Tensor:
    """
    <https://arxiv.org/abs/2312.02753>
    <https://arxiv.org/pdf/2006.09952.pdf>
    """

    match quantizer_noise_type:
        case "none":
            # FIXME: Hack for now
            noise = torch.empty(0)
        case "gaussian":
            noise = torch.randn_like(x, requires_grad=False) * cast(Tensor, noise_parameter)
        case "kumaraswamy":
            noise = generate_kumaraswamy_noise(
                torch.rand_like(x, requires_grad=False), cast(Tensor, noise_parameter)
            )

    match quantizer_type:
        case "none":
            return x + noise
        case "softround_alone":
            return softround(x, cast(Tensor, soft_round_temperature))
        case "softround":
            return softround(
                softround(x, cast(Tensor, soft_round_temperature)) + noise,
                cast(Tensor, soft_round_temperature),
            )
        case "ste":
            # From the forward point of view (i.e. entering into the torch.no_grad()), we have
            # y = softround(x) - softround(x) + round(x) = round(x). From the backward point of view
            # we have y = softround(x) meaning that dy / dx = d softround(x) / dx.
            y = softround(x, cast(Tensor, soft_round_temperature))
            with torch.no_grad():
                y = y - softround(x, cast(Tensor, soft_round_temperature)) + torch.round(x)
            return y
        case "hardround":
            return torch.round(x)
