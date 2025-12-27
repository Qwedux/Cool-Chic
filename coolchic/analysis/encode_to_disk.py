import os
import sys

os.chdir(os.path.dirname(os.getcwd()))
sys.path.append(os.getcwd())

import torch
from lossless.component.coolchic import (CoolChicEncoder,
                                         CoolChicEncoderParameter)
from coolchic.lossless.configs.config import args, str_args
from lossless.util.image_loading import load_image_as_tensor
from lossless.util.logger import TrainingLogger
from lossless.util.parsecli import (change_n_out_synth,
                                    get_coolchic_param_from_args)

torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision("high")

image_index = 0
use_color_regression = False
use_image_arm = True
encoder_gain = 64
multi_region_image_arm = True

im_path = args["input"][image_index]
im_tensor, c_bitdepths = load_image_as_tensor(im_path, device="cuda:0")
dataset = im_path.split("/")[-2]

# logger = TrainingLogger(
#     log_folder_path=args["LOG_PATH"],
#     image_name=f"{dataset}_" + im_path.split("/")[-1].split(".")[0],
# )
# logger.log_result(f"{str_args(args)}")
# logger.log_result(f"Processing image {im_path}")

encoder_param = CoolChicEncoderParameter(
    **get_coolchic_param_from_args(args, "lossless")
)
encoder_param.use_color_regression = use_color_regression
encoder_param.encoder_gain = encoder_gain
encoder_param.set_image_size((im_tensor.shape[2], im_tensor.shape[3]))
encoder_param.layers_synthesis = change_n_out_synth(
    encoder_param.layers_synthesis, 9 if use_color_regression else 6
)
encoder_param.use_image_arm = use_image_arm
encoder_param.multi_region_image_arm = multi_region_image_arm
coolchic = CoolChicEncoder(param=encoder_param)
coolchic.to_device("cuda:0")

coolchic.load_state_dict(torch.load("../logs_cluster/logs/full_runs/16_12_2025_ycocg_arm_smol_no_colreg_gain_test_multiimarm_kodak/trained_models/2025_12_15__20_35_44__trained_coolchic_kodak_kodim03_img_rate_2.5626282691955566.pth"))
print("Model loaded.")

