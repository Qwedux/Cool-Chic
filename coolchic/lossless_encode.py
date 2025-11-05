import os
import sys

sys.path.append(os.getcwd())
import torch
from lossless.component.coolchic import CoolChicEncoderParameter
from lossless.training.manager import (
    ImageEncoderManager,
)
from lossless.component.coolchic import CoolChicEncoder
from lossless.util.config import args, str_args
from lossless.util.parsecli import (
    change_n_out_synth,
    get_coolchic_param_from_args,
    get_manager_from_args,
)
from lossless.util.logger import TrainingLogger
from lossless.util.image_loading import load_image_as_tensor
from lossless.training.train import train
from lossless.training.loss import loss_function
from lossless.nnquant.quantizemodel import quantize_model

torch.autograd.set_detect_anomaly(True)

if len(sys.argv) < 4:
    print(
        "Usage: python3 lossless_encode.py <image_index> <color_space> <use_image_arm>"
    )
    print("<color_space> must be `YCoCg` or `RGB`")
    print("<use_image_arm> must be `true` or `false`")
    sys.exit(1)

# ==========================================================================================
# LOAD IMAGE
# ==========================================================================================
image_index = int(sys.argv[1])
color_space = sys.argv[2]
use_image_arm = sys.argv[3].lower() == "true"
assert sys.argv[3].lower() in [
    "true",
    "false",
], "<use_image_arm> must be `true` or `false`"
assert color_space in [
    "YCoCg",
    "RGB",
], f"Invalid color space {color_space}, must be YCoCg or RGB"

im_path = args["input"][image_index]
# im_path = "../datasets/synthetic/random_noise_256_256_white_gray.png"
im_tensor, c_bitdepths = load_image_as_tensor(
    im_path, device="cuda:0", color_space=color_space
)
# ==========================================================================================
# SETUP LOGGER
# ==========================================================================================
dataset_name = im_path.split("/")[-2]
logger = TrainingLogger(
    log_folder_path=args["LOG_PATH"],
    image_name=f"{dataset_name}_" + im_path.split("/")[-1].split(".")[0],
)
with open(args["network_yaml_path"], "r") as f:
    network_yaml = f.read()
logger.log_result(f"Network YAML configuration:\n{network_yaml}")
logger.log_result(f"{str_args(args)}")
logger.log_result(f"Processing image {im_path}")
logger.log_result(
    f"Using color space {color_space} with bitdepths {c_bitdepths.bitdepths}"
)
logger.log_result(f"Using image ARM: {use_image_arm}")
# ==========================================================================================
# LOAD PRESETS, COOLCHIC PARAMETERS
# ==========================================================================================
image_encoder_manager = ImageEncoderManager(**get_manager_from_args(args))
encoder_param = CoolChicEncoderParameter(
    **get_coolchic_param_from_args(args, "residue")
)
encoder_param.set_image_size((im_tensor.shape[2], im_tensor.shape[3]))
encoder_param.layers_synthesis = change_n_out_synth(
    encoder_param.layers_synthesis, args["output_dim_size"]
)
encoder_param.use_image_arm = use_image_arm
coolchic = CoolChicEncoder(param=encoder_param)
coolchic.to_device("cuda:0")
# ==========================================================================================
# TRAIN
# ==========================================================================================
if args["use_pretrained"]:
    coolchic.load_state_dict(torch.load(args["pretrained_model_path"]))
else:
    coolchic = train(
        model=coolchic,
        target_image=im_tensor,
        image_encoder_manager=image_encoder_manager,
        color_bitdepths=c_bitdepths,
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
# ==========================================================================================
# QUANTIZE AND EVALUATE
# ==========================================================================================
quantized_coolchic = CoolChicEncoder(param=encoder_param)
quantized_coolchic.to_device("cuda:0")
quantized_coolchic.set_param(coolchic.get_param())
quantized_coolchic = quantize_model(
    quantized_coolchic,
    im_tensor,
    image_encoder_manager,
    logger,
    color_bitdepths=c_bitdepths,
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
        channel_ranges=c_bitdepths,
    )
# logger.save_model(quantized_coolchic, predicted_priors_rates.loss.item())
logger.log_result(
    f"Final frame_encoder_manager state: {image_encoder_manager},\n"
    f"Rate per module: {rate_per_module},\n"
    f"Final results after quantization: {predicted_priors_rates}"
)
