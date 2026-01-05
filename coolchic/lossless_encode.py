import os
import sys

sys.path.append(os.getcwd())
import numpy as np
import torch
from lossless.component.coolchic import CoolChicEncoder
from lossless.configs.config import args, str_args
from lossless.io.decode_with_predictor import decode_with_predictor
from lossless.io.encode_with_predictor import encode_with_predictor
from lossless.io.encoding_interfaces.image_encoding_interface import \
    ImageEncodeDecodeInterface
from lossless.training.loss import loss_function
from lossless.training.manager import ImageEncoderManager
from lossless.training.train import train
from lossless.util.command_line_args_loading import load_args
from lossless.util.image_loading import load_image_as_tensor
from lossless.util.logger import TrainingLogger
from lossless.util.parsecli import get_coolchic_param_from_args

torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision("high")

# ==========================================================================================
# LOAD COMMAND LINE ARGS AND IMAGE
# ==========================================================================================
command_line_args = load_args()
im_path = args["input"][command_line_args.image_index]
im_tensor, colorspace_bitdepths = load_image_as_tensor(
    im_path, device="cuda:0", color_space=command_line_args.color_space
)
im_tensor = im_tensor[:,:64, :64] # for faster testing

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
# QUANTIZE AND EVALUATE
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

raw_synth_out, decoder_side_latent = coolchic.get_latents_raw_synth_out()
encode_decode_interface = ImageEncodeDecodeInterface(
    data=(torch.clone(im_tensor), torch.clone(raw_synth_out)), model=coolchic, ct_range=colorspace_bitdepths
)
bitstream, symbols_to_encode, prob_distributions_enc, channel_indices = encode_with_predictor(
    enc_dec_interface=encode_decode_interface,
    distribution="logistic",
    output_path="./test-workdir/encoder_size_test/coolchic_encoded_image.binary",
)
symbols_decoded, prob_distributions = decode_with_predictor(
    enc_dec_interface=encode_decode_interface,
    bitstream_path="./test-workdir/encoder_size_test/coolchic_encoded_image.binary",
    distribution="logistic",
)
logger.log_result("Encode-decode finished.")
symbols_enc_tensor = torch.tensor(symbols_to_encode)
symbols_dec_tensor = torch.tensor(symbols_decoded)
is_encode_decode_equal = torch.equal(symbols_enc_tensor, symbols_dec_tensor)
logger.log_result(f"Encode-decode equality check: {is_encode_decode_equal}")
logger.log_result(f"Rate Img bistream: {bitstream.nbytes * 8 / im_tensor.numel()}")

# ==========================================================================================
# SAVE MODEL
# ==========================================================================================
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
# model bpd is size of `coolchic_model_no_latents.pth` in bits divided by number of pixels in image
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
