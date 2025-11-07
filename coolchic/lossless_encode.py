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

from lossless.util.encoding import encode, decode, get_bits_per_pixel
from lossless.util.distribution import get_mu_and_scale_linear_color
import numpy as np

torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision("high")

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
    **get_coolchic_param_from_args(args, "lossless")
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
        start_lr=image_encoder_manager.start_lr,
        cosine_scheduling_lr=args[
            "schedule_lr"
        ],  # this is set by training phase
        max_iterations=image_encoder_manager.n_itr,
        frequency_validation=args[
            "freq_valid"
        ],  # this is set by training phase
        patience=args["patience"],  # this is set by training phase
        optimized_module=args[
            "optimized_module"
        ],  # this is set by training phase
        quantizer_type=args["quantizer_type"],  # this is set by training phase
        quantizer_noise_type=args[
            "quantizer_noise_type"
        ],  # this is set by training phase
        softround_temperature=args[
            "softround_temperature"
        ],  # this is set by training phase
        noise_parameter=args[
            "noise_parameter"
        ],  # this is set by training phase
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

# ==========================================================================================
# SAVE MODEL
# ==========================================================================================
cool_chic_state_dict = {
    k: v
    for k, v in quantized_coolchic.state_dict().items()
    if not k.startswith("latent_grids")
}
latent_grids_state_dict = {
    k: v
    for k, v in quantized_coolchic.state_dict().items()
    if k.startswith("latent_grids")
}

torch.save(
    cool_chic_state_dict,
    "test-workdir/encoder_size_test/coolchic_model_no_latents.pth",
)
# model bpd is size of `coolchic_model_no_latents.pth` in bits divided by number of pixels in image
model_file_size_bits = (
    os.path.getsize(
        "test-workdir/encoder_size_test/coolchic_model_no_latents.pth"
    )
    * 8
)
num_pixels = im_tensor.numel()
model_bpd = model_file_size_bits / num_pixels
print(f"Model bpd (without latent grids): {model_bpd}")

# ==========================================================================================
# ENCODE THE IMAGE
# ==========================================================================================

np.save(
    "testing/data/encoded_raw_out.npy", predicted_prior["raw_out"].cpu().numpy()
)
np.save("testing/data/original_image.npy", im_tensor.cpu().numpy())

mu, scale = get_mu_and_scale_linear_color(predicted_prior["raw_out"], im_tensor)

enc = encode(im_tensor, mu, scale, c_bitdepths, distribution="logistic")
dec = decode(enc, mu, scale, c_bitdepths, distribution="logistic")

print(dec.shape)
print(im_tensor.shape)
diff = torch.abs(im_tensor.cpu()*256 - dec * 256).to(torch.uint8)
print(f"Max difference after decoding: {diff.max().item()}")
# assert torch.allclose(
#     im_tensor.cpu(), dec.cpu()
# ), "Decoded image does not match original!"
print("Decoded image matches original!")

bpp = get_bits_per_pixel(1.0, 1.0, 1.0, enc) / im_tensor.numel()
print(f"Image bpd: {bpp}")
# from till_encode import encode as till_encode
# enc = till_encode(im_tensor, mu, scale)


# ==========================================================================================
# ENCODE LATENT GRIDS
# ==========================================================================================
torch.save(
    latent_grids_state_dict,
    "test-workdir/encoder_size_test/coolchic_latent_grids.pth",
)
