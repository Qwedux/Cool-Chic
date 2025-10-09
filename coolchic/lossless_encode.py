import os
import sys

sys.path.append(os.getcwd())
import torch
from lossless.component.coolchic import CoolChicEncoderParameter
from lossless.component.image import (
    FrameEncoderManager,
)
from lossless.component.coolchic import CoolChicEncoder
import numpy as np
import cv2
from lossless.util.config import args, str_args
from lossless.util.parsecli import (
    change_n_out_synth,
    get_coolchic_param_from_args,
    get_manager_from_args,
)
from lossless.util.misc import timestamp_string
from lossless.training.train import train
from lossless.nnquant.quantizemodel import quantize_model
from lossless.training.loss import loss_function
from lossless.util.logger import TrainingLogger
from lossless.util.color_transform import rgb_to_ycocg, ycocg_to_rgb
from lossless.util.image_loading import load_image_as_tensor
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python3 lossless_encode.py <image_index>")
    sys.exit(1)

image_index = int(sys.argv[1])
im_path = args["input"][image_index]
im_tensor = load_image_as_tensor(im_path, device="cuda:0")
dataset = im_path.split("/")[-2]

logger = TrainingLogger(
    log_folder_path=args["LOG_PATH"],
    image_name=f"{dataset}_" + im_path.split("/")[-1].split(".")[0],
)
logger.log_result(f"{str_args(args)}")
logger.log_result(f"Processing image {im_path}")

frame_encoder_manager = FrameEncoderManager(**get_manager_from_args(args))
encoder_param = CoolChicEncoderParameter(
    **get_coolchic_param_from_args(args, "residue")
)
encoder_param.set_image_size((im_tensor.shape[2], im_tensor.shape[3]))
encoder_param.layers_synthesis = change_n_out_synth(
    encoder_param.layers_synthesis, args["output_dim_size"]
)
coolchic = CoolChicEncoder(param=encoder_param)
coolchic.to_device("cuda:0")

if args["use_pretrained"]:
    coolchic.load_state_dict(torch.load(args["pretrained_model_path"]))
else:
    coolchic = train(
        model=coolchic,
        target_image=im_tensor,
        frame_encoder_manager=frame_encoder_manager,
        start_lr=args["start_lr"],
        lmbda=args["lmbda"],
        cosine_scheduling_lr=args["schedule_lr"],
        max_iterations=args["n_itr"],
        frequency_validation=args["freq_valid"],
        patience=args["patience"],
        optimized_module=args["optimized_module"],
        quantizer_type=args["quantizer_type"],
        quantizer_noise_type=args["quantizer_noise_type"],
        softround_temperature=args["softround_temperature"],
        noise_parameter=args["noise_parameter"],
        loss_latent_multiplier=1.0,
        logger=logger,
    )

quantized_coolchic = CoolChicEncoder(param=encoder_param)
quantized_coolchic.to_device("cuda:0")
quantized_coolchic.set_param(coolchic.get_param())
quantized_coolchic = quantize_model(
    quantized_coolchic, im_tensor, frame_encoder_manager, logger
)
rate_per_module, total_network_rate = quantized_coolchic.get_network_rate()
total_network_rate /= im_tensor.numel()
total_network_rate = float(total_network_rate)

with torch.no_grad():
    # Forward pass with no quantization noise
    predicted_prior = quantized_coolchic.forward(
        image=im_tensor,
        quantizer_noise_type="none",
        quantizer_type="hardround",
        AC_MAX_VAL=-1,
        flag_additional_outputs=False,
    )
    predicted_priors_rates = loss_function(
        predicted_prior,
        im_tensor,
        rate_mlp_bpd=total_network_rate,
        latent_multiplier=1.0,
    )
logger.save_model(quantized_coolchic, predicted_priors_rates.loss.item())
logger.log_result(
    f"Final frame_encoder_manager state: {frame_encoder_manager},\n"
    f"Rate per module: {rate_per_module},\n"
    f"Final results after quantization: {predicted_priors_rates}"
)
