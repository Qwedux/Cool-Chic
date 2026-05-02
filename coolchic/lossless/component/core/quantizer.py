# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

from __future__ import annotations

from typing import assert_never, cast

import torch
from lossless.component.core.types.quantization_noise_type import (
    POSSIBLE_QUANTIZATION_NOISE_TYPE, GaussianType, KumaraswamyType,
    NoQuantizationNoiseType)
from lossless.component.core.types.quantizier_type import (
    POSSIBLE_QUANTIZER_TYPE, HardroundType, NoQuantizerType,
    SoftroundAloneType, SoftroundType, STEType)
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


def quantize(
    x: Tensor,
    quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = KumaraswamyType(),
    quantizer_type: POSSIBLE_QUANTIZER_TYPE = SoftroundType(),
    soft_round_temperature: Tensor | None = None,
    noise_parameter: Tensor | None = None,
) -> Tensor:
    """
    <https://arxiv.org/abs/2312.02753>
    <https://arxiv.org/pdf/2006.09952.pdf>
    """

    match quantizer_noise_type:
        case NoQuantizationNoiseType():
            # FIXME: Hack for now
            noise = torch.empty(0)
        case GaussianType():
            noise = torch.randn_like(x, requires_grad=False) * cast(Tensor, noise_parameter)
        case KumaraswamyType():
            noise = generate_kumaraswamy_noise(
                torch.rand_like(x, requires_grad=False), cast(Tensor, noise_parameter)
            )
        case _:
            assert_never(quantizer_noise_type)

    match quantizer_type:
        case NoQuantizerType():
            return x + noise
        case SoftroundAloneType():
            return softround(x, cast(Tensor, soft_round_temperature))
        case SoftroundType():
            return softround(
                softround(x, cast(Tensor, soft_round_temperature)) + noise,
                cast(Tensor, soft_round_temperature),
            )
        case STEType():
            # From the forward point of view (i.e. entering into the torch.no_grad()), we have
            # y = softround(x) - softround(x) + round(x) = round(x). From the backward point of view
            # we have y = softround(x) meaning that dy / dx = d softround(x) / dx.
            y = softround(x, cast(Tensor, soft_round_temperature))
            with torch.no_grad():
                y = y - softround(x, cast(Tensor, soft_round_temperature)) + torch.round(x)
            return y
        case HardroundType():
            return torch.round(x)
        case _:
            assert_never(quantizer_type)