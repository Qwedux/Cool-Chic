import glob
import os

from lossless.component.core.arm_image import ImageARMParameter

if os.path.exists("/itet-stor/jparada/net_scratch/"):
    BASE_PATH = "/itet-stor/jparada/net_scratch/"
    DATASET_PATH = f"{BASE_PATH}datasets/kodak/"
    TEST_WORKDIR = f"{BASE_PATH}Cool-Chic/coolchic/test-workdir/"
    LOG_PATH = "/home/jparada/logs/"
    NETWORK_YAML_PATH = f"{BASE_PATH}Cool-Chic/cfg/network_architecture.yaml"
else:
    BASE_PATH = f"{os.getcwd()}/../"
    DATASET_PATH = f"{BASE_PATH}datasets/clic2024/"
    TEST_WORKDIR = f"{BASE_PATH}coolchic/test-workdir/"
    LOG_PATH = f"{BASE_PATH}logs/"
    NETWORK_YAML_PATH = f"{BASE_PATH}cfg/network_architecture.yaml"

IMAGE_PATHS = sorted(
    glob.glob(f"{DATASET_PATH}*.png"),
    # key=lambda x: int(os.path.basename(x).split(".")[0][len("kodim") :]),
)

args = {
    # paths
    "BASE_PATH": BASE_PATH,
    "DATASET_PATH": DATASET_PATH,
    "TEST_WORKDIR": TEST_WORKDIR,
    "LOG_PATH": LOG_PATH,
    "input": IMAGE_PATHS,
    "output": TEST_WORKDIR + "output",
    "workdir": TEST_WORKDIR,
    "network_yaml_path": NETWORK_YAML_PATH,
    "experiment_name": "2026_01_17_speed_test_lossless",

    "print_detailed_archi": False,
    "print_detailed_struct": False,
    # config file paths
    # encoder side
    # decoder side
    "layers_synthesis_lossless": "24-1-1-linear-relu,X-1-1-linear-none,X-3-3-residual-relu,X-3-3-residual-none",
    "arm_lossless": "16,2", #dim arm, n_layers
    "arm_lossless_hidden_layer_dim": 8,
    "arm_image_params": ImageARMParameter(context_size=8, n_hidden_layers=2, hidden_layer_dim=6),
    "use_color_regression": False,
    "n_ft_per_res_lossless": "1,1,1,1,1,1,1",
    "ups_k_size_lossless": 8,
    "ups_preconcat_k_size_lossless": 7,
    # training preset
    "preset": "fnlic",
    "pretrained_model_path": "../logs/full_runs/2026_01_05_default_name/trained_models/2026_01_05__20_55_36__trained_coolchic_kodak_kodim01_img_rate_4.001727104187012.pth",
    "use_pretrained": False,
    "quantize_model": True,
}

def str_args(args: dict) -> str:
    included_keys = args.keys()
    s = "Arguments:\n"
    for k in included_keys:
        s += f"  {k}: {args[k]}\n"
    return s

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
