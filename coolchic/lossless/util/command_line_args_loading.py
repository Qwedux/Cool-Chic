from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, TypeAlias

import tyro
from lossless.component.coolchic import (CoolChicComputingMode, UseFullModel,
                                         UseImageARMOnly, UseUntilSynthesis)
from lossless.util.colorspace import PossibleColorspace, YCoCg

# Type alias for CommandLineArgs field names
COMMAND_LINE_ARGS_NAMES = Literal[
    "image_index",
    "encoder_gain",
    "color_space",
    "experiment_name",
    "multiarm_setup",
    "computing_mode",
    "profile_training",
    "profile_output_dir",
]

ExperimentName: TypeAlias = str
EncoderGain: TypeAlias = int
ImageIndex: TypeAlias = int
MultiArmSetup: TypeAlias = str

@dataclass
class CommandLineArgs:
    image_index: ImageIndex = field(default_factory=lambda: ImageIndex(0))
    encoder_gain: EncoderGain = field(default_factory=lambda: EncoderGain(64))
    color_space: PossibleColorspace = field(default_factory=lambda: YCoCg())
    experiment_name: ExperimentName = field(
        default_factory=lambda: ExperimentName(datetime.today().strftime("%Y_%m_%d_default_name"))
    )
    multiarm_setup: MultiArmSetup = field(
        default_factory=lambda: MultiArmSetup("1x1")
    )  # e.g., "2x2" for 2 rows and 2 columns
    computing_mode: CoolChicComputingMode = field(default_factory=UseFullModel)
    profile_training: bool = False
    profile_output_dir: str | None = None


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


def load_args(notebook_overrides: dict[COMMAND_LINE_ARGS_NAMES, Any] = {}) -> CommandLineArgs:
    """
    Returns the config.
    - If running in a Notebook, returns manual defaults (prevents crashing).
    - If running in CLI, parses sys.argv flags via Tyro.
    """
    if _is_notebook():
        default_args = CommandLineArgs()
        # Override any defaults with notebook_overrides
        for key, value in notebook_overrides.items():
            if not hasattr(default_args, key):
                continue
            current = getattr(default_args, key)
            # Union of marker dataclasses: isinstance against default instance class is too narrow.
            if key == "computing_mode" and isinstance(
                value, (UseFullModel, UseImageARMOnly, UseUntilSynthesis)
            ):
                setattr(default_args, key, value)
            elif isinstance(value, type(current)):
                setattr(default_args, key, value)
        return default_args
    else:
        # Tyro automatically reads global sys.argv
        if len(notebook_overrides) > 0:
            warnings.warn(
                "notebook_overrides provided but running outside notebook; ignoring them."
            )
        return tyro.cli(CommandLineArgs)
