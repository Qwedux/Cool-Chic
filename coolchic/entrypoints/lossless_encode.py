import os
import sys
import time

sys.path.append(os.getcwd())
from typing import cast

import torch
from lossless.component.coolchic import CoolChicEncoder
from lossless.configs.config import args
from lossless.training.loss import loss_function
from lossless.training.manager import ImageEncoderManager
from lossless.training.train import train
from lossless.util.command_line_args_loading import load_args
from lossless.util.device import CpuDevice, CudaZeroDevice
from lossless.util.image import ImageSize
from lossless.util.image_loading import load_image_as_tensor
from lossless.util.logger import TrainingLogger
from lossless.util.parsecli import get_coolchic_param_from_args

torch.set_float32_matmul_precision("high")
# torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# ==========================================================================================
# LOAD COMMAND LINE ARGS AND IMAGE
# ==========================================================================================
command_line_args = load_args()
job_index = command_line_args.image_index
print(f"Encoding job {job_index} started at {time.time()}")
im_path = args.input[command_line_args.image_index]

im_tensor, colorspace_bitdepths = load_image_as_tensor(
    im_path, device=CudaZeroDevice(), color_space=command_line_args.color_space
)

# ==========================================================================================
# LOAD PRESETS, COOLCHIC PARAMETERS
# ==========================================================================================
multi_arm_setup: tuple[int, int] = cast(
    tuple[int, int], tuple(map(int, command_line_args.multiarm_setup.split("x")))
)
image_encoder_manager = ImageEncoderManager(
    preset_name=args.preset,
    colorspace_bitdepths=colorspace_bitdepths,
    multi_region_image_arm_setup=multi_arm_setup,
)

encoder_param = get_coolchic_param_from_args(
    args=args,
    image_size=ImageSize(width=im_tensor.shape[3], height=im_tensor.shape[2]),
    encoder_gain=command_line_args.encoder_gain,
)
coolchic = CoolChicEncoder(param=encoder_param, computing_mode=command_line_args.computing_mode, device=CudaZeroDevice())
coolchic.to_device(CudaZeroDevice())

# ==========================================================================================
# SETUP LOGGER
# ==========================================================================================
dataset_name = im_path.split("/")[-2]
logger = TrainingLogger(
    log_folder_path=args.LOG_PATH,
    image_name=f"{dataset_name}_" + im_path.split("/")[-1].split(".")[0] + f"_job_{job_index}",
    debug_mode=image_encoder_manager.n_itr < 1000,
    experiment_name=command_line_args.experiment_name,
    # disable console print for now
    console_print=True,
)
logger.log_result(f"Preset: {image_encoder_manager.preset.pretty_string()}")
logger.log_result(f"{args}")
logger.log_result(f"Job index: {job_index}")
logger.log_result(f"Image index: {command_line_args.image_index}")
logger.log_result(f"Processing image {im_path}")
logger.log_result(
    f"Using color space {command_line_args.color_space} with bitdepths {image_encoder_manager.colorspace_bitdepths.bitdepths}"
)
logger.log_result(f"Using encoder gain: {command_line_args.encoder_gain}")
logger.log_result(f"Using multi-region image ARM: {command_line_args.multiarm_setup}")
logger.log_result(f"Using color regression: {args.use_color_regression}")
logger.log_result(f"Total training iterations: {image_encoder_manager.n_itr}")
with torch.no_grad():
    coolchic.to_device(CpuDevice())
    logger.log_result(f"Total MAC per pixel: {coolchic.get_total_mac_per_pixel()}")
    logger.log_result(coolchic.str_complexity())
    coolchic.to_device(CudaZeroDevice())
# ==========================================================================================
# TRAIN
# ==========================================================================================
if args.use_pretrained:
    coolchic.load_state_dict(torch.load(args.pretrained_model_path))
else:
    coolchic = train(
        model=coolchic,
        target_image=im_tensor,
        image_encoder_manager=image_encoder_manager,
        logger=logger,
    )
logger.log_result(
    f"Training completed in {image_encoder_manager.total_training_time_sec:.2f} seconds"
)
# ==========================================================================================
# QUANTIZE AND SAVE MODEL
# ==========================================================================================
rate_per_module, total_network_rate = coolchic.get_network_rate()
total_network_rate = float(total_network_rate) / im_tensor.numel()

with torch.no_grad():
    # Forward pass with no quantization noise
    predicted_prior = coolchic.forward(
        image=im_tensor,
        quantizer_noise_type="none",
        quantizer_type="hardround",
        AC_MAX_VAL=-1,
    )
    predicted_priors_rates = loss_function(
        predicted_prior,
        im_tensor,
        rate_mlp_bpd=total_network_rate,
        colorspace_bitdepths=colorspace_bitdepths,
    )

logger.save_model(coolchic, predicted_priors_rates.loss.item())
logger.log_result(
    f"Final frame_encoder_manager state: {image_encoder_manager},\n"
    f"Rate per module: {rate_per_module},\n"
    f"Final results after quantization: {predicted_priors_rates}"
)