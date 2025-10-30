import os
import sys

sys.path.append(os.getcwd())
import torch
from lossless.component.coolchic import CoolChicEncoderParameter
from lossless.component.image import (
    ImageEncoderManager,
)
from lossless.component.coolchic import CoolChicEncoder
from lossless.util.config import args, str_args
from lossless.util.parsecli import (
    change_n_out_synth,
    get_coolchic_param_from_args,
    get_manager_from_args,
)
from lossless.training.train import train
from lossless.nnquant.quantizemodel import quantize_model
from lossless.training.loss import loss_function
from lossless.util.logger import TrainingLogger
from lossless.util.image_loading import load_image_as_tensor
from lossless.util.distribution import get_mu_and_scale_linear_color
from lossless.util.encoding import encode, decode, get_bits_per_pixel
import numpy as np

torch.autograd.set_detect_anomaly(True)

if len(sys.argv) < 3:
    print("Usage: python3 lossless_encode.py <image_index> <color_space>")
    print("<color_space> must be `YCoCg` or `RGB`")
    sys.exit(1)

image_index = int(sys.argv[1])
color_space = sys.argv[2]
assert color_space in [
    "YCoCg",
    "RGB",
], f"Invalid color space {color_space}, must be YCoCg or RGB"

# im_path = args["input"][image_index]
im_path = "../datasets/synthetic/random_noise_64_64_black_gray.png"
im_tensor, c_bitdepths = load_image_as_tensor(
    im_path, device="cuda:0", color_space=color_space
)

encoder_param = CoolChicEncoderParameter(
    **get_coolchic_param_from_args(args, "residue")
)
encoder_param.set_image_size((im_tensor.shape[2], im_tensor.shape[3]))
encoder_param.layers_synthesis = change_n_out_synth(
    encoder_param.layers_synthesis, args["output_dim_size"]
)
coolchic = CoolChicEncoder(param=encoder_param)
if args["use_pretrained"]:
    coolchic.load_state_dict(torch.load(args["pretrained_model_path"]))
else:
    exit("Pretrained model required for encoding.")
coolchic.to_device("cuda:0")

quantized_coolchic = CoolChicEncoder(param=encoder_param)
quantized_coolchic.to_device("cuda:0")
quantized_coolchic.set_param(coolchic.get_param())
quantized_coolchic = quantize_model(
    quantized_coolchic,
    im_tensor,
    None,
    None,
    color_bitdepths=c_bitdepths,
)
# torch.save(coolchic.image_arm.state_dict(), "image_arm_before_quantization.pth")

rate_per_module, total_network_rate = quantized_coolchic.get_network_rate()
total_network_rate += 10621 * 8 # bits for encoding the image_arm
total_network_rate /= im_tensor.numel()

with torch.no_grad():
    # Forward pass with no quantization noise
    predicted_prior = quantized_coolchic.forward(
        image=im_tensor,
        quantizer_noise_type="none",
        quantizer_type="hardround",
        AC_MAX_VAL=-1,
    )
    # second_predicted_prior = quantized_coolchic.encode_to_bitstream(
    #     image=im_tensor,
    #     quantizer_noise_type="none",
    #     quantizer_type="hardround",
    #     AC_MAX_VAL=-1,
    # )
    # if torch.allclose(
    #     predicted_prior["raw_out"],
    #     second_predicted_prior["raw_out"],
    # ):
    #     print("Quantized model forward matches original model forward.")
    # else:
    #     print("Warning: Quantized model forward does not match original model forward.")
    #     # print the first value that differs and its indices
    #     diff = predicted_prior["raw_out"] - second_predicted_prior["raw_out"]
    #     indices = torch.nonzero(diff)
    #     first_diff = indices[0]
    #     print(
    #         f"First difference at index {first_diff.tolist()}: "
    #         f"{predicted_prior['raw_out'][tuple(first_diff.tolist())]} vs "
    #         f"{second_predicted_prior['raw_out'][tuple(first_diff.tolist())]}"
    #     )
    
    predicted_prior["latent_bpd"] = torch.tensor(0.0022786459885537624)
    predicted_priors_rates = loss_function(
        predicted_prior,
        im_tensor,
        rate_mlp_bpd=total_network_rate,
        latent_multiplier=1.0,
        channel_ranges=c_bitdepths,
    )
print(
    f"Rate per module: {rate_per_module},\n"
    f"Final results after quantization: {predicted_priors_rates}"
)

np.save("testing/data/encoded_raw_out.npy", predicted_prior["raw_out"].cpu().numpy())
np.save("testing/data/original_image.npy", im_tensor.cpu().numpy())

mu, scale = get_mu_and_scale_linear_color(predicted_prior["raw_out"], im_tensor)
enc = encode(im_tensor, mu, scale)
dec = decode(enc, mu, scale)
assert torch.allclose(im_tensor.cpu(), dec.cpu()), "Decoded image does not match original!"
print("Decoded image matches original!")

bpp = get_bits_per_pixel(64, 64, 3, enc)
print(f"Final image bpp: {bpp}")

