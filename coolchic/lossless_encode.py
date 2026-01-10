import os
import sys

sys.path.append(os.getcwd())
import copy

import numpy as np
import torch
from lossless.component.coolchic import CoolChicEncoder
from lossless.configs.config import args, str_args
from lossless.io.decode_with_predictor import decode_with_predictor
from lossless.io.encode_with_predictor import encode_with_predictor
from lossless.io.encoding_interfaces.image_encoding_interface import \
    ImageEncodeDecodeInterface
from lossless.io.encoding_interfaces.latent_encoding_interface import \
    LatentEncodeDecodeInterface
from lossless.training.loss import loss_function
from lossless.training.manager import ImageEncoderManager
from lossless.training.train import train
from lossless.util.color_transform import LatentBitdepths
from lossless.util.command_line_args_loading import load_args
from lossless.util.image_loading import load_image_as_tensor
from lossless.util.logger import TrainingLogger
from lossless.util.parsecli import get_coolchic_param_from_args

torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision("high")
# torch._logging.set_logs(graph_code=True)

# ==========================================================================================
# LOAD COMMAND LINE ARGS AND IMAGE
# ==========================================================================================
command_line_args = load_args()
im_path = args["input"][command_line_args.image_index]
im_tensor, colorspace_bitdepths = load_image_as_tensor(
    im_path, device="cuda:0", color_space=command_line_args.color_space
)
# im_tensor = im_tensor[:,:64, :64] # for faster testing

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
coolchic.to_device("cuda:0")
# ==========================================================================================
# SETUP LOGGER
# ==========================================================================================
dataset_name = im_path.split("/")[-2]
logger = TrainingLogger(
    log_folder_path=args["LOG_PATH"],
    image_name=f"{dataset_name}_" + im_path.split("/")[-1].split(".")[0],
    debug_mode=image_encoder_manager.n_itr < 1000,
    experiment_name=command_line_args.experiment_name,
)
with open(args["network_yaml_path"], "r") as f:
    network_yaml = f.read()
logger.log_result(f"Preset: {image_encoder_manager.preset.pretty_string()}")
# logger.log_result(f"Network YAML configuration:\n{network_yaml}")
logger.log_result(f"{str_args(args)}")
logger.log_result(f"Processing image {im_path}")
logger.log_result(
    f"Using color space {command_line_args.color_space} with bitdepths {image_encoder_manager.colorspace_bitdepths.bitdepths}"
)
logger.log_result(f"Using image ARM: {command_line_args.use_image_arm}")
logger.log_result(f"Using encoder gain: {command_line_args.encoder_gain}")
logger.log_result(f"Using multi-region image ARM: {args['multi_region_image_arm']}")
logger.log_result(f"Using color regression: {args['use_color_regression']}")
logger.log_result(f"Total training iterations: {image_encoder_manager.n_itr}")
with torch.no_grad():
    logger.log_result(f"Total MAC per pixel: {coolchic.get_total_mac_per_pixel()}")
    logger.log_result(coolchic.str_complexity())
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
        logger=logger,
    )
logger.log_result(
    f"Training completed in {image_encoder_manager.total_training_time_sec:.2f} seconds"
)
# ==========================================================================================
# QUANTIZE AND SAVE MODEL
# ==========================================================================================
rate_per_module, total_network_rate = coolchic.get_network_rate()
if command_line_args.use_image_arm:
    arm_params = list(coolchic.image_arm.parameters())
    arm_params_bits = sum(p.numel() for p in arm_params) * 32
    total_network_rate += arm_params_bits
total_network_rate = float(total_network_rate) / im_tensor.numel()

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
        rate_mlp_bpd=total_network_rate,
        colorspace_bitdepths=colorspace_bitdepths,
    )

logger.save_model(coolchic, predicted_priors_rates.loss.item())
logger.log_result(
    f"Final frame_encoder_manager state: {image_encoder_manager},\n"
    f"Rate per module: {rate_per_module},\n"
    f"Final results after quantization: {predicted_priors_rates}"
)

# ==========================================================================================
# FULL ENCODE-DECODE TO BITSTREAM
# ==========================================================================================

coolchic.to_device("cpu")
im_tensor = im_tensor.to("cpu")
with torch.no_grad():
    raw_synth_out, decoder_side_latent = coolchic.get_latents_raw_synth_out(AC_MAX_VAL=31)

# first do image
enc_dec_im_interface = ImageEncodeDecodeInterface(
    data=(torch.clone(im_tensor), torch.clone(raw_synth_out)),
    model=coolchic,
    ct_range=colorspace_bitdepths,
)
bitstream_im, im_symbols_pre_encoding, _, _ = encode_with_predictor(
    enc_dec_interface=enc_dec_im_interface,
    logger=logger,
    distribution="logistic",
    output_path=None,
)
im_symbols_post_encoding, prob_distributions = decode_with_predictor(
    enc_dec_interface=enc_dec_im_interface,
    bitstream=bitstream_im,
    bitstream_path=None,
    distribution="logistic",
)
logger.log_result("Image encode-decode finished.")
is_im_encode_decode_equal = torch.equal(
    torch.tensor(im_symbols_pre_encoding), torch.tensor(im_symbols_post_encoding)
)
logger.log_result(f"Image encode-decode equality check: {is_im_encode_decode_equal}")
logger.log_result(f"Rate Img bistream: {bitstream_im.nbytes * 8 / im_tensor.numel()}")

# # second do latents
# enc_dec_latent_interface = LatentEncodeDecodeInterface(
#     data=copy.deepcopy(decoder_side_latent), model=coolchic, ct_range=LatentBitdepths()
# )
# bitstream_latent, latent_symbols_pre_encoding, _, _ = encode_with_predictor(
#     enc_dec_interface=enc_dec_latent_interface,
#     distribution="laplace",
#     output_path=None,
#     logger=logger,
# )
# latent_symbols_post_encoding, prob_distributions_dec = decode_with_predictor(
#     enc_dec_interface=enc_dec_latent_interface,
#     distribution="laplace",
#     bitstream=bitstream_latent,
#     bitstream_path=None,
# )
# logger.log_result("Latent encode-decode finished.")
# is_latent_encode_decode_equal = torch.equal(
#     torch.tensor(latent_symbols_pre_encoding),
#     torch.tensor(latent_symbols_post_encoding),
# )
# logger.log_result(f"Latent encode-decode equality check: {is_latent_encode_decode_equal}")
# logger.log_result(f"Rate Latent bistream: {bitstream_latent.nbytes * 8 / im_tensor.numel()}")
# logger.log_result(
#     f"Total image+latent bpd rate: {(bitstream_im.nbytes + bitstream_latent.nbytes) * 8 / im_tensor.numel()}"
# )
