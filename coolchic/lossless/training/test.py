# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


from __future__ import annotations

import torch
from lossless.component.coolchic import CoolChicEncoder
from lossless.training.loss import LossFunctionOutput, loss_function
from lossless.training.manager import ImageEncoderManager


@torch.no_grad()
def test(
    model: CoolChicEncoder,
    frame: torch.Tensor,
    image_encoder_manager: ImageEncoderManager,
) -> LossFunctionOutput:
    model.eval()
    frame_encoder_out = model.forward(
        image=frame,
        quantizer_noise_type="none",
        quantizer_type="hardround",
        AC_MAX_VAL=-1,
    )

    loss_fn_output = loss_function(
        frame_encoder_out,
        frame,
        colorspace_bitdepths=image_encoder_manager.colorspace_bitdepths,
    )
    model.train()

    return loss_fn_output
