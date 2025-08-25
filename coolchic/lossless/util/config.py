import os
import glob

DATASET_PATH = f"{os.getcwd()}/../datasets/kodak"
IMAGE_PATHS = sorted(
    glob.glob(f"{DATASET_PATH}/*.png"),
    key=lambda x: int(os.path.basename(x).split(".")[0][len("kodim") :]),
)
TEST_WORKDIR = f"{os.getcwd()}/test-workdir/"
PATH_COOL_CHIC_CFG = f"{os.getcwd()}/../cfg/"


args = {
    # not in config files
    "input": IMAGE_PATHS[0],
    "output": TEST_WORKDIR + "output",
    "workdir": TEST_WORKDIR,
    "lmbda": 1e-3,
    "job_duration_min": -1,
    "print_detailed_archi": False,
    "print_detailed_struct": False,
    # config file paths
    # encoder side
    "start_lr": 1e-2,
    "n_itr": 1000,
    "n_train_loops": 100,
    "preset": "debug",
    # decoder side
    "layers_synthesis_residue": "16-1-linear-relu,X-1-linear-none,X-3-residual-relu,X-3-residual-none",
    "arm_residue": "8,2",
    "n_ft_per_res_residue": "1,1,1,1,1,1,1",
    "ups_k_size_residue": 8,
    "ups_preconcat_k_size_residue": 7,
    # training preset
    "patience": 10,
    "schedule_lr": True,
    "freq_valid": 10,
    "optimized_module": ["all"],
    "quantizer_type": "softround",
    "quantizer_noise_type": "gaussian",
    "softround_temperature": (0.3, 0.1),
    "noise_parameter": (0.25, 0.1),
    # Other presets
    "quantize_model": True
}


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
