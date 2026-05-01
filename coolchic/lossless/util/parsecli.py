from __future__ import annotations

from collections.abc import Sequence

import torch
from lossless.component.coolchic import CoolChicEncoderParameter
from lossless.configs.config import Args
from lossless.util.image import ImageSize


def parse_synthesis_layers(layers_synthesis: str) -> Sequence[str]:
    parsed_layer_synth = [x for x in layers_synthesis.split(",") if x != ""]
    return parsed_layer_synth


def change_n_out_synth(layers_synth: Sequence[str], n_out: int) -> Sequence[str]:
    return [lay.replace("X", str(n_out)) for lay in layers_synth]


def get_coolchic_param_from_args(
    args: Args,
    image_size: ImageSize,
    encoder_gain: int,
) -> CoolChicEncoderParameter:
    layers_synthesis = parse_synthesis_layers(args.layers_synthesis)
    return CoolChicEncoderParameter(
        layers_synthesis=change_n_out_synth(
            layers_synthesis, 9 if args.use_color_regression else 6
        ),
        n_ft_per_res=args.n_ft_per_res,
        image_arm_parameters=args.arm_image_params,
        arm_latent_parameters=args.arm_latent_parameters,
        encoder_gain=encoder_gain,
        ups_k_size=args.ups_k_size,
        ups_preconcat_k_size=args.ups_preconcat_k_size,
        latent_freq_precision=args.latent_freq_precision,
        use_color_regression=args.use_color_regression,
        img_size=image_size,
    )
