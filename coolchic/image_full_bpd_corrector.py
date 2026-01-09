import os
import sys

sys.path.append(os.getcwd())

import copy

import numpy as np
import torch
from lossless.component.coolchic import CoolChicEncoder
from lossless.configs.config import args
from lossless.io.encode_with_predictor import encode_with_predictor
from lossless.io.encoding_interfaces.image_encoding_interface import \
    ImageEncodeDecodeInterface
from lossless.training.loss import loss_function
from lossless.training.manager import ImageEncoderManager
from lossless.util.command_line_args_loading import load_args
from lossless.util.image_loading import load_image_as_tensor
from lossless.util.parsecli import get_coolchic_param_from_args

torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision("high")
torch.use_deterministic_algorithms(True)

# ==========================================================================================
# LOAD COMMAND LINE ARGS AND IMAGE
# ==========================================================================================
command_line_args = load_args(
    notebook_overrides={
        "image_index": 4,
        "encoder_gain": 64,
        "color_space": "YCoCg",
        "use_image_arm": True,
    }
)
im_path = args["input"][command_line_args.image_index]
im_tensor, colorspace_bitdepths = load_image_as_tensor(
    im_path, device="cpu", color_space=command_line_args.color_space
)
# ==========================================================================================
# LOAD PRESETS, COOLCHIC PARAMETERS
# ==========================================================================================
image_encoder_manager = ImageEncoderManager(
    preset_name=args["preset"], colorspace_bitdepths=colorspace_bitdepths
)

encoder_param = get_coolchic_param_from_args(
    args,
    "lossless",
    image_size=(im_tensor.shape[2], im_tensor.shape[3]),
    use_image_arm=command_line_args.use_image_arm,
    encoder_gain=command_line_args.encoder_gain,
)
coolchic = CoolChicEncoder(param=encoder_param)
coolchic.load_state_dict(
    torch.load(
        "../logs_cluster/logs/07_01_2026_YCoCg_arm_smol_no_color_regression_gain_test_multiarm_Kodak/trained_models/2026_01_07__22_17_02__trained_coolchic_kodak_kodim05_img_rate_3.6469767093658447.pth"
    )
)

coolchic.eval()
coolchic.to_device("cpu")
with torch.no_grad():
    raw_synth_out, decoder_side_latent = coolchic.get_latents_raw_synth_out(AC_MAX_VAL=-1)
    print("raw_synth_out:", raw_synth_out.shape)
    print("flat_decoder_side_latent:", len(decoder_side_latent))

with torch.no_grad():
    # Forward pass with no quantization noise
    predicted_prior = coolchic.forward(
        image=im_tensor,
        quantizer_noise_type="none",
        quantizer_type="hardround",
        AC_MAX_VAL=-1,
        flag_additional_outputs=False,
    )
    predicted_priors_rates = loss_function(
        predicted_prior,
        im_tensor,
        rate_mlp_bpd=0.0,
        colorspace_bitdepths=colorspace_bitdepths,
    )
    print(predicted_priors_rates)

reduced_decoder_side_latent = copy.deepcopy(decoder_side_latent)

reduced_im_tensor = torch.clone(im_tensor)
reduced_raw_synth_out = torch.clone(raw_synth_out)
encode_decode_interface = ImageEncodeDecodeInterface(
    data=(torch.clone(reduced_im_tensor), torch.clone(reduced_raw_synth_out)), model=coolchic, ct_range=colorspace_bitdepths
)
bitstream, symbols_to_encode, prob_distributions_enc, channel_indices = encode_with_predictor(
    enc_dec_interface=encode_decode_interface,
    distribution="logistic",
    output_path=None,
    logger=None,
)
print(bitstream.nbytes * 8 / im_tensor.numel())
print("Encoded", len(symbols_to_encode), "symbols.")