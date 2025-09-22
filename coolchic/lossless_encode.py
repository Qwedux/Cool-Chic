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
from lossless.util.config import (
    TEST_WORKDIR,
    # IMAGE_PATHS,
    args,
)
import random
from lossless.util.parsecli import (
    change_n_out_synth,
    get_coolchic_param_from_args,
    get_manager_from_args,
)
from lossless.util.misc import clean_workdir, timestamp_string
from lossless.training.train import train
# from lossless.util.encoding import encode, get_bits_per_pixel
# from lossless.util.distribution import get_scale, get_mu_scale, get_mu_and_scale_linear_color
from lossless.nnquant.quantizemodel import quantize_model
from lossless.training.loss import loss_function

if len(sys.argv) < 2:
    print("Usage: python3 lossless_encode.py <image_index>")
    sys.exit(1)

image_index = int(sys.argv[1])
print(f"Processing image kodim_{'0' + str(image_index+1) if image_index < 9 else image_index+1}.png")

frame_encoder_manager = FrameEncoderManager(**get_manager_from_args(args))
# im_path = IMAGE_PATHS[random.randint(0, len(IMAGE_PATHS))]
im_path = args["input"][image_index]
print(im_path)
im = cv2.imread(filename=im_path)
assert im is not None, f"Failed to read image {args['input']}"
im = im[:, :, ::-1]  # Convert BGR to RGB
im_tensor = torch.from_numpy(im.copy()).float() / 255.0  # Normalize to [0, 1]
im_tensor = im_tensor.permute((2, 0, 1))[None,]  # Change to CxHxW
im_tensor = im_tensor.to("cuda" if torch.cuda.is_available() else "cpu")

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
    )


quantized_coolchic = CoolChicEncoder(param=encoder_param)
quantized_coolchic.to_device("cuda:0")
quantized_coolchic.set_param(coolchic.get_param())
quantized_coolchic = quantize_model(
    quantized_coolchic, im_tensor, frame_encoder_manager
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
        predicted_prior, im_tensor, latent_multiplier=1.0
    )

# mu, scale = get_mu_and_scale_linear_color(predicted_prior["raw_out"], im_tensor)
# encoded_bytes = encode(im_tensor, mu, scale)
# assert predicted_priors_rates.rate_latent_bpd is not None
# image_bpd = get_bits_per_pixel(768, 512, 3, encoded_bytes)
assert predicted_priors_rates.rate_img_bpd is not None
assert predicted_priors_rates.rate_latent_bpd is not None

all_rates = {
    "total_bpd": predicted_priors_rates.rate_img_bpd
    + predicted_priors_rates.rate_latent_bpd
    + float(total_network_rate),
    "image_bpd": predicted_priors_rates.rate_img_bpd,
    "latent_bpd": predicted_priors_rates.rate_latent_bpd,
    "network_bpd": total_network_rate,
}
print(all_rates)
# save all_rates to a text file
with open(f"{TEST_WORKDIR}/{timestamp_string()}_{im_path.split('/')[-1]}_rates.txt", "a") as f:
    f.write(f"{all_rates}\n")


torch.save(
    quantized_coolchic.state_dict(),
    f"{TEST_WORKDIR}/{timestamp_string()}_trained_coolchic_{im_path.split('/')[-1]}_img_rate_{all_rates['total_bpd']}.pth",
)

