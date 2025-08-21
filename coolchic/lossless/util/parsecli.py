from typing import List, Dict, Any

def parse_synthesis_layers(layers_synthesis: str) -> List[str]:
    """The layers of the synthesis are presented in as a coma-separated string.
    This simply splits up the different substrings and return them.

    Args:
        layers_synthesis (str): Command line argument for the synthesis.

    Returns:
        List[str]: List of string where the i-th element described the i-th
            synthesis layer
    """
    parsed_layer_synth = [x for x in layers_synthesis.split(",") if x != ""]

    assert parsed_layer_synth, (
        "Synthesis should have at least one layer, found nothing. \n"
        f"--layers_synthesis={layers_synthesis} does not work!\n"
        "Try something like 32-1-linear-relu,X-1-linear-none,"
        "X-3-residual-relu,X-3-residual-none"
    )

    return parsed_layer_synth


def parse_n_ft_per_res(n_ft_per_res: str) -> list[int]:
    """The number of feature per resolution is a coma-separated string.
    This simply splits up the different substrings and return them.

    Args:
        n_ft_per_res (str): Something like "1,1,1,1,1,1,1" for 7 latent grids
        with different resolution and 1 feature each.

    Returns:
        List[int]: The i-th element is the number of features for the i-th
        latent, i.e. the latent of a resolution (H / 2^i, W / 2^i).
    """

    n_ft_per_res_int = [int(x) for x in n_ft_per_res.split(",") if x != ""]
    # assert set(n_ft_per_res) == {
    #     1
    # }, f"--n_ft_per_res should only contains 1. Found {n_ft_per_res}"
    return n_ft_per_res_int


def parse_arm_archi(arm: str) -> Dict[str, int]:
    """The arm is described as <dim_arm>,<n_hidden_layers_arm>.
    Split up this string to return the value as a dict.

    Args:
        arm (str): Command line argument for the ARM.

    Returns:
        Dict[str, int]: The ARM architecture
    """
    assert len(arm.split(",")) == 2, (
        f"--arm format should be X,Y." f" Found {arm}"
    )

    dim_arm, n_hidden_layers_arm = [int(x) for x in arm.split(",")]
    arm_param = {"dim_arm": dim_arm, "n_hidden_layers_arm": n_hidden_layers_arm}
    return arm_param


def get_coolchic_param_from_args(
    args: dict,
    coolchic_enc_name: str,
) -> Dict[str, Any]:
    layers_synthesis = parse_synthesis_layers(
        args[f"layers_synthesis_{coolchic_enc_name}"]
    )
    n_ft_per_res = parse_n_ft_per_res(args[f"n_ft_per_res_{coolchic_enc_name}"])

    coolchic_param = {
        "layers_synthesis": layers_synthesis,
        "n_ft_per_res": n_ft_per_res,
        "ups_k_size": args[f"ups_k_size_{coolchic_enc_name}"],
        "ups_preconcat_k_size": args[
            f"ups_preconcat_k_size_{coolchic_enc_name}"
        ],
    }

    # Add ARM parameters
    coolchic_param.update(parse_arm_archi(args[f"arm_{coolchic_enc_name}"]))

    return coolchic_param

def change_n_out_synth(layers_synth: List[str], n_out: int) -> List[str]:
        """Change the number of output features in the list of strings
        describing the synthesis architecture. It replaces "X" with n_out. E.g.

        From [8-1-linear-relu,X-1-linear-none,X-3-residual-none]
        To   [8-1-linear-relu,2-1-linear-none,2-3-residual-none]

        If n_out = 2

        Args:
            layers_synth (List[str]): List of strings describing the different
                synthesis layers
            n_out (int): Number of desired output.

        Returns:
            List[str]: List of strings with the proper number of output features.
        """
        return [lay.replace("X", str(n_out)) for lay in layers_synth]

def get_manager_from_args(args: dict) -> Dict[str, Any]:
    """Perform some check on the argparse object used to collect the command
    line parameters. Return a dictionary ready to be plugged into the
    ``FrameEncoderManager`` constructor.

    Args:
        args (argparse.Namespace): Command-line argument parser.

    Returns:
        Dict[str, Any]: Dictionary ready to be plugged into the
            ``FrameEncoderManager`` constructor.
    """
    frame_encoder_manager = {
        "preset_name": args["preset"],
        "start_lr": args["start_lr"],
        "lmbda": args["lmbda"],
        "n_loops": args["n_train_loops"],
        "n_itr": args["n_itr"],
    }
    return frame_encoder_manager