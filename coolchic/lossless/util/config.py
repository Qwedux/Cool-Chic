import os
import glob

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

    "lmbda": 1e-3,
    "job_duration_min": -1,
    "print_detailed_archi": False,
    "print_detailed_struct": False,
    # config file paths
    # encoder side
    "start_lr": 1e-2,
    "n_itr": 50000,
    "n_train_loops": 1,
    "preset": "fnlic",
    # decoder side
    "layers_synthesis_residue": "24-1-linear-relu,X-1-linear-none,X-3-residual-relu,X-3-residual-none",
    "arm_residue": "16,2",
    "arm_image_context_size": "8",
    "n_ft_per_res_residue": "1,1,1,1,1,1,1",
    "ups_k_size_residue": 8,
    "ups_preconcat_k_size_residue": 7,
    "output_dim_size": 9,
    # training preset
    "patience": 5000,
    "schedule_lr": True,
    "freq_valid": 100,
    "optimized_module": ["all"],
    "quantizer_type": "softround",
    "quantizer_noise_type": "kumaraswamy",
    "softround_temperature": (0.3, 0.1),
    "noise_parameter": (0.25, 0.1),
    "pretrained_model_path": "../logs/trained_models/2025_10_27__16_52_28__trained_coolchic_synthetic_random_noise_256_256_white_gray_img_rate_7.170324325561523.pth",
    "use_pretrained": False,
    # Other presets
    "quantize_model": True,
}

def str_args(args: dict) -> str:
    included_keys = [
        "DATASET_PATH",
        "TEST_WORKDIR",
        "LOG_PATH",
        "workdir",
        "lmbda",
        "job_duration_min",
        "print_detailed_archi",
        "print_detailed_struct",

        "start_lr",
        "n_itr",
        "n_train_loops",
        "preset",
        # decoder side
        "layers_synthesis_residue",
        "arm_residue",
        "arm_image_context_size",
        "n_ft_per_res_residue",
        "ups_k_size_residue",
        "ups_preconcat_k_size_residue",
        "output_dim_size",
        # training preset
        "patience",
        "schedule_lr",
        "freq_valid",
        "optimized_module",
        "quantizer_type",
        "quantizer_noise_type",
        "softround_temperature",
        "noise_parameter",
        # Other presets
        "quantize_model",
    ]
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
