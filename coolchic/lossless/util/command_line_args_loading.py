import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import tyro


@dataclass
class CommandLineArgs:
    # No default value -> Required argument in CLI
    # Has a default value -> Optional argument in CLI

    image_index: int = field(default_factory=lambda: 0)
    encoder_gain: int = field(default_factory=lambda: 64)
    # Literal enforces "YCoCg" or "RGB".
    color_space: Literal["YCoCg", "RGB"] = field(default_factory=lambda: "YCoCg")
    # Handling booleans: --use-image-arm / --no-use-image-arm
    use_image_arm: bool = field(default_factory=lambda: True)
    experiment_name: str = field(
        default_factory=lambda: datetime.today().strftime("%Y_%m_%d_default_name")
    )
    multiarm_setup: str = field(
        default_factory=lambda: "1x1"
    )  # e.g., "2x2" for 2 rows and 2 columns


def _is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__  # type: ignore
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def load_args(notebook_overrides: dict = {}) -> CommandLineArgs:
    """
    Returns the config.
    - If running in a Notebook, returns manual defaults (prevents crashing).
    - If running in CLI, parses sys.argv flags via Tyro.
    """
    if _is_notebook():
        default_args = CommandLineArgs()
        # Override any defaults with notebook_overrides
        for key, value in notebook_overrides.items():
            if hasattr(default_args, key) and isinstance(value, type(getattr(default_args, key))):
                # set only if the attribute exists
                setattr(default_args, key, value)
        return default_args
    else:
        # Tyro automatically reads global sys.argv
        if len(notebook_overrides) > 0:
            warnings.warn(
                "notebook_overrides provided but running outside notebook; ignoring them."
            )
        return tyro.cli(CommandLineArgs)
