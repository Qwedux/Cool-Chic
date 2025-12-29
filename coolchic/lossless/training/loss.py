# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


from dataclasses import dataclass

import torch
from lossless.component.coolchic import CoolChicEncoderOutput
from lossless.util.color_transform import ColorBitdepths
from lossless.util.distribution import weak_colorar_rate


@dataclass(kw_only=True)
class LossFunctionOutput:
    """Output for Encoder.loss_function"""

    # ----- This is the important output
    # Optional to allow easy inheritance by EncoderLogs
    loss: torch.Tensor  # The RD cost to optimize

    # Any other data required to compute some logs, stored inside a dictionary
    rate_nn_bpd: float  # Rate associated to the neural networks [bpd]
    rate_latent_bpd: float  # Rate associated to the latent      [bpd]
    rate_img_bpd: float  # Rate associated to the image          [bpd]

    def __str__(self) -> str:
        return (
            f"Loss: {self.loss.item()}, "
            f"Rate NN: {self.rate_nn_bpd}, "
            f"Rate Latent: {self.rate_latent_bpd}, "
            f"Rate Img: {self.rate_img_bpd}"
        )


def loss_function(
    encoder_out: CoolChicEncoderOutput,
    img_tensor: torch.Tensor,
    channel_ranges: ColorBitdepths,
    rate_mlp_bpd: float = 0.0,
    latent_multiplier: float = 1.00,
) -> LossFunctionOutput:
    img_rates = weak_colorar_rate(
        encoder_out["mu"], encoder_out["scale"], img_tensor, channel_ranges
    )
    img_bpd = img_rates.sum() / img_tensor.numel()
    loss = img_bpd + rate_mlp_bpd + encoder_out["latent_bpd"] * latent_multiplier
    return LossFunctionOutput(
        loss=loss,
        rate_nn_bpd=rate_mlp_bpd,
        rate_latent_bpd=float((encoder_out["latent_bpd"]).detach().item()),
        rate_img_bpd=float(img_bpd.detach().item()),
    )
