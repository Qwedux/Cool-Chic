import os
import sys

sys.path.append(os.getcwd())
import numpy as np
import torch
from lossless.component.coolchic import (CoolChicEncoder,
                                         CoolChicEncoderParameter)
from lossless.nnquant.quantizemodel import quantize_model
from lossless.training.loss import loss_function
from lossless.training.manager import ImageEncoderManager
from lossless.training.train import train
from lossless.util.image_loading import load_image_as_tensor
from lossless.util.logger import TrainingLogger
from lossless.util.parsecli import (change_n_out_synth,
                                    get_coolchic_param_from_args,
                                    get_manager_from_args)
from till_encode import decode, encode, get_bits_per_pixel

from coolchic.lossless.configs.config import args, str_args

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
im_tensor, c_bitdepths = load_image_as_tensor(
    im_path, device="cuda:0", color_space=color_space
)

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
# SETUP LOGGER
# ==========================================================================================
dataset_name = im_path.split("/")[-2]
logger = TrainingLogger(
    log_folder_path=args["LOG_PATH"],
    image_name=f"{dataset_name}_" + im_path.split("/")[-1].split(".")[0],
    debug_mode=image_encoder_manager.n_itr < 1000,
    experiment_name=args["experiment_name"],
)
logger.log_result(f"Testing model {args['pretrained_model_path']}")
# ==========================================================================================
# TRAIN
# ==========================================================================================
if args["use_pretrained"]:
    coolchic.load_state_dict(torch.load(args["pretrained_model_path"]))
else:
    raise NotImplementedError("Training not implemented in this script.")
# ==========================================================================================
# QUANTIZE AND EVALUATE
# ==========================================================================================
# technically we don't need quantization when working with uncompressed model
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
        colorspace_bitdepths=c_bitdepths,
        use_color_regression=args["use_color_regression"],
    )
logger.log_result(
    f"Rate per module: {rate_per_module},\n"
    f"Final results after quantization: {predicted_priors_rates}"
)

# # ==========================================================================================
# # SAVE MODEL
# # ==========================================================================================
# cool_chic_state_dict = {
#     k: v
#     for k, v in quantized_coolchic.state_dict().items()
#     if not k.startswith("latent_grids")
# }
# latent_grids_state_dict = {
#     k: v
#     for k, v in quantized_coolchic.state_dict().items()
#     if k.startswith("latent_grids")
# }

# torch.save(
#     cool_chic_state_dict,
#     "test-workdir/encoder_size_test/coolchic_model_no_latents.pth",
# )
# # model bpd is size of `coolchic_model_no_latents.pth` in bits divided by number of pixels in image
# model_file_size_bits = (
#     os.path.getsize(
#         "test-workdir/encoder_size_test/coolchic_model_no_latents.pth"
#     )
#     * 8
# )
# num_pixels = im_tensor.numel()
# model_bpd = model_file_size_bits / num_pixels
# logger.log_result(f"Rate NN (without latent grids): {model_bpd}")

# # ==========================================================================================
# # ENCODE-DECODE THE IMAGE // Takes ~few minutes!!!
# # ==========================================================================================

# np.save(
#     "testing/data/encoded_raw_out.npy", predicted_prior["raw_out"].cpu().numpy()
# )
# np.save("testing/data/original_image.npy", im_tensor.cpu().numpy())

# mu, scale = get_mu_and_scale_linear_color(predicted_prior["raw_out"], im_tensor)

# logger.log_result("Starting encoding...")
# bitstream, probs_logistic = encode(
#     im_tensor,
#     mu,
#     scale,
#     c_bitdepths,
#     distribution="logistic",
#     output_path="./test-workdir/encoder_size_test/coolchic_encoded_image.binary",
# )
# logger.log_result("Starting decoding...")
# decoded_image, probs_logistic = decode(
#     "./test-workdir/encoder_size_test/coolchic_encoded_image.binary",
#     mu,
#     scale,
#     c_bitdepths,
#     distribution="logistic",
# )
# logger.log_result("Encode-decode finished.")
# # get filesize of encoded file
# encoded_file_size = os.path.getsize(
#     "./test-workdir/encoder_size_test/coolchic_encoded_image.binary"
# )
# image_bpd = encoded_file_size * 8 / im_tensor.numel()
# logger.log_result(f"Rate Img: {image_bpd}")
# assert torch.allclose(
#     im_tensor.cpu(), decoded_image.cpu()
# ), "Decoded image does not match original!"
# logger.log_result("Decoded image matches original!")


# # ==========================================================================================
# # ENCODE LATENT GRIDS
# # ==========================================================================================
# torch.save(
#     latent_grids_state_dict,
#     "test-workdir/encoder_size_test/coolchic_latent_grids.pth",
# )

# latent_bpd = predicted_priors_rates.rate_latent_bpd
# logger.log_result(f"Rate Latent: {latent_bpd}")
# logger.log_result(f"Loss: {image_bpd + model_bpd + latent_bpd}")
