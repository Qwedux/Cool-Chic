from __future__ import annotations

import glob
import os
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from lossless.component.core.arm import ArmParameter
from lossless.component.core.arm_image import (ImageARMParameter,
                                               MultiImageArmDescriptor)
from lossless.util.image import ImageHeight, ImageWidth

if os.path.exists("/itet-stor/jparada/net_scratch/"):
    BASE_PATH = "/itet-stor/jparada/net_scratch/"
    DATASET_PATH = f"{BASE_PATH}datasets/kodak/"
    TEST_WORKDIR = f"{BASE_PATH}Cool-Chic/coolchic/test-workdir/"
    LOG_PATH = "/home/jparada/logs/"
    NETWORK_YAML_PATH = f"{BASE_PATH}Cool-Chic/cfg/network_architecture.yaml"
else:
    BASE_PATH = f"{os.getcwd()}/../"
    DATASET_PATH = f"{BASE_PATH}datasets/kodak/"
    TEST_WORKDIR = f"{BASE_PATH}coolchic/test-workdir/"
    LOG_PATH = f"{BASE_PATH}logs/"
    NETWORK_YAML_PATH = f"{BASE_PATH}cfg/network_architecture.yaml"

IMAGE_PATHS: Sequence[str] = sorted(
    glob.glob(f"{DATASET_PATH}*.png"),
)

def default_image_arm_parameters(use_color_regression: bool) -> ImageARMParameter:
    if use_color_regression:
        synthesis_out_params_per_channel = [2, 3, 4]
    else:
        synthesis_out_params_per_channel = [2, 2, 2]
    return ImageARMParameter(
        context_size=8,
        n_hidden_layers=2,
        hidden_layer_dim=6,
        synthesis_out_params_per_channel=synthesis_out_params_per_channel,
        use_color_regression=use_color_regression,
        multi_region_image_arm_specification=MultiImageArmDescriptor(
            expert_indices=[],
            image_height=ImageHeight(1),
            image_width=ImageWidth(1),
            routing_grid=torch.zeros(1, 1),
        )

    )


@dataclass(frozen=True, kw_only=True)
class Args:
    # paths
    BASE_PATH: str
    DATASET_PATH: str
    TEST_WORKDIR: str
    LOG_PATH: str
    input: Sequence[str]
    output: str
    workdir: str
    print_detailed_archi: bool
    print_detailed_struct: bool
    layers_synthesis: str
    arm_latent_parameters: ArmParameter
    arm_image_params: ImageARMParameter
    use_color_regression: bool
    n_ft_per_res: Sequence[int]
    ups_k_size: int
    ups_preconcat_k_size: int
    preset: str
    pretrained_model_path: str
    use_pretrained: bool
    quantize_model: bool
    latent_freq_precision: int

_use_color_regression = False

args = Args(
    BASE_PATH=BASE_PATH,
    DATASET_PATH=DATASET_PATH,
    TEST_WORKDIR=TEST_WORKDIR,
    LOG_PATH=LOG_PATH,
    input=IMAGE_PATHS,
    output=TEST_WORKDIR + "output",
    workdir=TEST_WORKDIR,
    print_detailed_archi=False,
    print_detailed_struct=False,
    layers_synthesis="24-1-1-linear-relu,X-1-1-linear-none,X-3-3-residual-relu,X-3-3-residual-none",
    arm_latent_parameters=ArmParameter(context_size=16, n_hidden_layers=2, hidden_layer_dim=8),
    arm_image_params=default_image_arm_parameters(use_color_regression=_use_color_regression),
    use_color_regression=_use_color_regression,
    n_ft_per_res=[1,1,1,1,1,1,1],
    ups_k_size=8,
    ups_preconcat_k_size=7,
    preset="measure_speed",
    pretrained_model_path="../logs/full_runs/2026_01_05_default_name/trained_models/2026_01_05__20_55_36__trained_coolchic_kodak_kodim01_img_rate_4.001727104187012.pth",
    use_pretrained=False,
    quantize_model=True,
    latent_freq_precision=12,
)

start_print = (
    "\n\n"
    "*----------------------------------------------------------------------------------------------------------*\n"
    "|                                                                                                          |\n"
    "|                                                                                                          |\n"
    "|       ,gggg,                                                                                             |\n"
    '|     ,88"""Y8b,                           ,dPYb,                             ,dPYb,                       |\n'
    "|    d8\"     `Y8                           IP'`Yb                             IP'`Yb                       |\n"
    "|   d8'   8b  d8                           I8  8I                             I8  8I      gg               |\n"
    "|  ,8I    \"Y88P'                           I8  8'                             I8  8'      \"\"               |\n"
    "|  I8'             ,ggggg,      ,ggggg,    I8 dP      aaaaaaaa        ,gggg,  I8 dPgg,    gg     ,gggg,    |\n"
    '|  d8             dP"  "Y8ggg  dP"  "Y8ggg I8dP       """"""""       dP"  "Yb I8dP" "8I   88    dP"  "Yb   |\n'
    "|  Y8,           i8'    ,8I   i8'    ,8I   I8P                      i8'       I8P    I8   88   i8'         |\n"
    "|  `Yba,,_____, ,d8,   ,d8'  ,d8,   ,d8'  ,d8b,_                   ,d8,_    _,d8     I8,_,88,_,d8,_    _   |\n"
    '|    `"Y8888888 P"Y8888P"    P"Y8888P"    8P\'"Y88                  P""Y8888PP88P     `Y88P""Y8P""Y8888PP   |\n'
    "|                                                                                                          |\n"
    "|                                                                                                          |\n"
    "| version 4.1.0, July 2025                                                              © 2023-2025 Orange |\n"
    "*----------------------------------------------------------------------------------------------------------*\n"
)
