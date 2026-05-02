# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

from __future__ import annotations

import typing
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from lossless.component.core.types.quantization_noise_type import (
    POSSIBLE_QUANTIZATION_NOISE_TYPE, GaussianType, KumaraswamyType,
    NoQuantizationNoiseType)
from lossless.component.core.types.quantizier_type import (
    POSSIBLE_QUANTIZER_TYPE, SoftroundType, STEType)


@dataclass(frozen=True)
class OptimizeAll:
    pass


@dataclass(frozen=True)
class OptimizeArm:
    pass


@dataclass(frozen=True)
class OptimizeArmImage:
    pass


@dataclass(frozen=True)
class OptimizeUpsampling:
    pass


@dataclass(frozen=True)
class OptimizeSynthesis:
    pass


@dataclass(frozen=True)
class OptimizeLatent:
    pass


MODULE_TO_OPTIMIZE = (
    OptimizeAll
    | OptimizeArm
    | OptimizeArmImage
    | OptimizeUpsampling
    | OptimizeSynthesis
    | OptimizeLatent
)


@dataclass
class TrainerPhase:
    lr: float = 1e-2
    max_itr: int = 10000
    freq_valid: int = 100
    patience: int = 1000
    quantize_model: bool = False
    schedule_lr: bool = False
    softround_temperature: Tuple[float, float] = (0.3, 0.3)
    noise_parameter: Tuple[float, float] = (2.0, 1.0)
    quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = KumaraswamyType()
    quantizer_type: POSSIBLE_QUANTIZER_TYPE = SoftroundType()
    optimized_module: Sequence[MODULE_TO_OPTIMIZE] = (OptimizeAll(),)

    def pretty_string(self) -> str:
        """Return a pretty string describing a warm-up phase"""

        s = f'{f"{self.lr:1.2e}":^{14}}|'
        s += f"{' '.join([type(module).__name__ for module in self.optimized_module]):^{20}}|"
        s += f"{self.max_itr:^{9}}|"
        s += f"{self.patience:^{16}}|"
        s += f"{self.freq_valid:^{13}}|"
        s += f"{self.quantize_model:^{13}}|"
        s += f"{self.schedule_lr:^{13}}|"

        softround_str = ", ".join([f"{x:1.1e}" for x in self.softround_temperature])
        s += f'{f"{softround_str}":^{18}}|'

        noise_str = ", ".join([f"{x:1.2f}" for x in self.noise_parameter])
        s += f'{f"{noise_str}":^{14}}|'
        return s

    @classmethod
    def _pretty_string_column_name(cls) -> str:
        """Return the name of the column aligned with the pretty_string function"""
        s = f'{"Learn rate":^{14}}|'
        s += f'{"Module optimized":^{20}}|'
        s += f'{"Max itr":^{9}}|'
        s += f'{"Patience [itr]":^{16}}|'
        s += f'{"Valid [itr]":^{13}}|'
        s += f'{"Quantize NN":^{13}}|'
        s += f'{"Schedule lr":^{13}}|'
        s += f'{"Softround Temp":^{18}}|'
        s += f'{"Noise":^{14}}|'
        return s

    @classmethod
    def _vertical_line_array(cls) -> str:
        """Return a string made of "-" and "+" matching the columns
        of the print detailed above"""
        s = "-" * 14 + "+"
        s += "-" * 20 + "+"
        s += "-" * 9 + "+"
        s += "-" * 16 + "+"
        s += "-" * 13 + "+"
        s += "-" * 13 + "+"
        s += "-" * 13 + "+"
        s += "-" * 18 + "+"
        s += "-" * 14 + "+"
        return s


@dataclass
class WarmupPhase:
    candidates: (
        int  # Keep the first <candidates> best systems at the beginning of this warmup phase
    )
    training_phase: TrainerPhase

    def pretty_string(self) -> str:
        """Return a pretty string describing a warm-up phase"""
        s = f"|{self.candidates:^{14}}|"
        s += f"{self.training_phase.pretty_string()}"
        return s

    @classmethod
    def _pretty_string_column_name(cls) -> str:
        """Return the name of the column aligned with the pretty_string function"""
        s = f'|{"Candidates":^{14}}|'
        s += f"{TrainerPhase._pretty_string_column_name()}"
        return s


@dataclass
class Warmup:
    phases: List[WarmupPhase] = field(default_factory=lambda: [])

    def _get_total_warmup_iterations(self) -> int:
        return sum([phase.candidates * phase.training_phase.max_itr for phase in self.phases])


@dataclass
class Preset:
    preset_name: str
    # Dummy empty training phases and warm-up
    warmup: Warmup = field(default_factory=lambda: Warmup())  # All the warm-up phases
    training_phases: List[TrainerPhase] = field(
        default_factory=lambda: []
    )  # All the post-warm-up training phases

    def __post_init__(self):
        # Check that we do quantize the model at least once during the training
        flag_quantize_model = False
        for training_phase in self.training_phases:
            if training_phase.quantize_model:
                flag_quantize_model = True

        # Ignore this assertion if there is no self.training_phases described
        assert flag_quantize_model or len(self.training_phases) == 0, (
            f"The selected preset ({self.preset_name}) does not include "
            f" a training phase with neural network quantization.\n"
            f"{self.pretty_string()}"
        )

    def _get_total_training_iterations(self) -> int:
        """Return the total number of iterations for the whole warm-up."""
        return (
            sum([phase.max_itr for phase in self.training_phases])
            + self.warmup._get_total_warmup_iterations()
        )

    def pretty_string(self) -> str:
        """Return a pretty string describing a warm-up phase"""
        s = f"Preset: {self.preset_name:<10}\n"
        s += "-------\n"

        s += "\nWarm-up\n"
        s += "-------\n"
        s += "+" + "-" * 14 + "+" + TrainerPhase._vertical_line_array() + "\n"
        s += WarmupPhase._pretty_string_column_name() + "\n"
        s += "+" + "-" * 14 + "+" + TrainerPhase._vertical_line_array() + "\n"
        for warmup_phase in self.warmup.phases:
            s += warmup_phase.pretty_string() + "\n"
        s += "+" + "-" * 14 + "+" + TrainerPhase._vertical_line_array() + "\n"

        s += "\nMain training\n"
        s += "-------------\n"
        s += "+" + "-" * 14 + "+" + TrainerPhase._vertical_line_array() + "\n"
        s += f'|{"Phase index":^14}|{TrainerPhase._pretty_string_column_name()}\n'
        s += "+" + "-" * 14 + "+" + TrainerPhase._vertical_line_array() + "\n"
        for idx, training_phase in enumerate(self.training_phases):
            s += f"|{idx:^14}|{training_phase.pretty_string()}\n"
        s += "+" + "-" * 14 + "+" + TrainerPhase._vertical_line_array() + "\n"

        s += "\nMaximum number of iterations (warm-up / training / total):"
        warmup_max_itr = self.warmup._get_total_warmup_iterations()
        training_max_itr = self._get_total_training_iterations()
        total_max_itr = warmup_max_itr + training_max_itr
        s += f"{warmup_max_itr:^8} / " f"{training_max_itr:^8} / " f"{total_max_itr:^8}\n"
        return s


class PresetFNLIC(Preset):
    def __init__(self):
        super().__init__(preset_name="fnlic")
        # 1st stage: with soft round and kumaraswamy noise
        self.training_phases: List[TrainerPhase] = [
            TrainerPhase(
                lr=1e-2,
                max_itr=120000,
                freq_valid=100,
                patience=4000,
                quantize_model=False,
                schedule_lr=True,
                softround_temperature=(0.3, 0.3),
                noise_parameter=(2.0, 2.0),
                quantizer_noise_type=KumaraswamyType(),
                quantizer_type=SoftroundType(),
                optimized_module=[OptimizeAll()],
            )
        ]

        self.training_phases.append(
            TrainerPhase(
                lr=1.0e-4,
                max_itr=1500,
                freq_valid=100,
                patience=1500,
                optimized_module=[OptimizeAll()],
                schedule_lr=True,
                quantizer_type=STEType(),
                quantizer_noise_type=NoQuantizationNoiseType(),
                # This is only used to parameterize the backward of the quantization
                softround_temperature=(1e-4, 1e-4),
                noise_parameter=(1.0, 1.0),  # not used since quantizer type is "ste"
                quantize_model=True,  # ! This is an important parameter
            )
        )

        self.training_phases.append(
            TrainerPhase(
                lr=1e-4,
                max_itr=1000,
                freq_valid=10,
                patience=50,
                quantize_model=False,
                schedule_lr=False,
                softround_temperature=(1e-4, 1e-4),
                noise_parameter=(1.0, 1.0),
                quantizer_noise_type=NoQuantizationNoiseType(),
                quantizer_type=STEType(),
                optimized_module=[OptimizeLatent()],  # ! Only fine tune the latent
            )
        )

        # 5 candidates, then 2 then 1
        self.warmup = Warmup(
            [
                WarmupPhase(
                    candidates=16,
                    training_phase=TrainerPhase(
                        lr=1e-2,
                        max_itr=1000,
                        freq_valid=100,
                        patience=100000,
                        quantize_model=False,
                        schedule_lr=False,
                        softround_temperature=(0.3, 0.3),
                        noise_parameter=(2.0, 2.0),
                        quantizer_noise_type=KumaraswamyType(),
                        quantizer_type=SoftroundType(),
                        optimized_module=[OptimizeAll()],
                    ),
                ),
                WarmupPhase(
                    candidates=8,
                    training_phase=TrainerPhase(
                        lr=1e-2,
                        max_itr=1000,
                        freq_valid=100,
                        patience=100000,
                        quantize_model=False,
                        schedule_lr=False,
                        softround_temperature=(0.3, 0.3),
                        noise_parameter=(2.0, 2.0),
                        quantizer_noise_type=KumaraswamyType(),
                        quantizer_type=SoftroundType(),
                        optimized_module=[OptimizeAll()],
                    ),
                ),
                WarmupPhase(
                    candidates=4,
                    training_phase=TrainerPhase(
                        lr=1e-2,
                        max_itr=1000,
                        freq_valid=100,
                        patience=100000,
                        quantize_model=False,
                        schedule_lr=False,
                        softround_temperature=(0.3, 0.3),
                        noise_parameter=(2.0, 2.0),
                        quantizer_noise_type=KumaraswamyType(),
                        quantizer_type=SoftroundType(),
                        optimized_module=[OptimizeAll()],
                    ),
                ),
                WarmupPhase(
                    candidates=2,
                    training_phase=TrainerPhase(
                        lr=1e-2,
                        max_itr=1000,
                        freq_valid=100,
                        patience=100000,
                        quantize_model=False,
                        schedule_lr=False,
                        softround_temperature=(0.3, 0.3),
                        noise_parameter=(2.0, 2.0),
                        quantizer_noise_type=KumaraswamyType(),
                        quantizer_type=SoftroundType(),
                        optimized_module=[OptimizeAll()],
                    ),
                ),
            ]
        )


class PresetC3xIntra(Preset):
    def __init__(
        self,
        start_lr: float = 1e-2,
        itr_main_training: int = 100000,
    ):
        super().__init__(preset_name="c3x_intra")
        # 1st stage: with soft round and quantization noise
        self.training_phases: List[TrainerPhase] = [
            TrainerPhase(
                lr=start_lr,
                max_itr=itr_main_training,
                patience=5000,
                optimized_module=[OptimizeAll()],
                schedule_lr=True,
                quantizer_type=SoftroundType(),
                quantizer_noise_type=GaussianType(),
                softround_temperature=(0.3, 0.1),
                noise_parameter=(0.25, 0.1),
                # quantize_model=True,  # ! This is an important parameter
            ),
            # Stage with STE then network quantization
            TrainerPhase(
                lr=1.0e-4,
                max_itr=1500,
                patience=1500,
                optimized_module=[OptimizeAll()],
                schedule_lr=True,
                quantizer_type=STEType(),
                quantizer_noise_type=NoQuantizationNoiseType(),
                # This is only used to parameterize the backward of the quantization
                softround_temperature=(1e-4, 1e-4),
                noise_parameter=(
                    1.0,
                    1.0,
                ),  # not used since quantizer type is "ste"
                quantize_model=True,  # ! This is an important parameter
            ),
            # # Re-tune the latent
            # TrainerPhase(
            #     lr=1.0e-4,
            #     max_itr=1000,
            #     patience=50,
            #     quantizer_type="ste",
            #     quantizer_noise_type="none",
            #     optimized_module=["latent"],  # ! Only fine tune the latent
            #     freq_valid=10,
            #     softround_temperature=(1e-4, 1e-4),
            #     noise_parameter=(1.0, 1.0),     # not used since quantizer type is "ste"
            # ),
        ]

        self.warmup = Warmup(
            [
                WarmupPhase(
                    candidates=5,
                    training_phase=TrainerPhase(
                        lr=start_lr,
                        max_itr=400,
                        freq_valid=400,
                        patience=100000,
                        quantize_model=False,
                        schedule_lr=False,
                        softround_temperature=(0.3, 0.3),
                        noise_parameter=(2.0, 2.0),
                        quantizer_noise_type=KumaraswamyType(),
                        quantizer_type=SoftroundType(),
                        optimized_module=[OptimizeAll()],
                    ),
                ),
                WarmupPhase(
                    candidates=2,
                    training_phase=TrainerPhase(
                        lr=start_lr,
                        max_itr=400,
                        freq_valid=400,
                        patience=100000,
                        quantize_model=False,
                        schedule_lr=False,
                        softround_temperature=(0.3, 0.3),
                        noise_parameter=(2.0, 2.0),
                        quantizer_noise_type=KumaraswamyType(),
                        quantizer_type=SoftroundType(),
                        optimized_module=[OptimizeAll()],
                    ),
                ),
            ]
        )


class PresetC3xInter(Preset):
    def __init__(
        self,
        start_lr: float = 1e-2,
        itr_main_training: int = 100000,
    ):
        super().__init__(preset_name="c3x_inter")
        # 1st stage: with soft round and quantization noise
        self.training_phases: List[TrainerPhase] = [
            TrainerPhase(
                lr=start_lr,
                max_itr=itr_main_training,
                patience=5000,
                optimized_module=[OptimizeAll()],
                schedule_lr=True,
                quantizer_type=SoftroundType(),
                quantizer_noise_type=GaussianType(),
                softround_temperature=(0.3, 0.1),
                noise_parameter=(0.25, 0.1),
                quantize_model=True,  # ! This is an important parameter
            ),
        ]

        self.warmup = Warmup(
            [
                WarmupPhase(
                    candidates=2,
                    training_phase=TrainerPhase(
                        lr=start_lr,
                        max_itr=600,
                        freq_valid=600,
                        patience=100000,
                        quantize_model=False,
                        schedule_lr=False,
                        softround_temperature=(0.3, 0.3),
                        noise_parameter=(2.0, 2.0),
                        quantizer_noise_type=KumaraswamyType(),
                        quantizer_type=SoftroundType(),
                        optimized_module=[OptimizeAll()],
                    ),
                )
            ]
        )


class PresetDebug(Preset):
    """Very fast training schedule, should only be used to ensure that the code works properly!"""

    def __init__(
        self,
    ):
        super().__init__(preset_name="debug")
        self.training_phases: List[TrainerPhase] = [
            TrainerPhase(
                lr=1e-2,
                max_itr=1000,
                freq_valid=10,
                patience=100000,
                quantize_model=False,
                schedule_lr=True,
                softround_temperature=(0.3, 0.1),
                noise_parameter=(0.25, 0.1),
                quantizer_noise_type=KumaraswamyType(),
                quantizer_type=SoftroundType(),
                optimized_module=[OptimizeAll()],
            ),
            TrainerPhase(
                lr=1e-4,
                max_itr=10,
                freq_valid=100,
                patience=100,
                quantize_model=True,
                schedule_lr=False,
                softround_temperature=(1e-4, 1e-4),
                noise_parameter=(1.0, 1.0),
                quantizer_noise_type=NoQuantizationNoiseType(),
                quantizer_type=STEType(),
                optimized_module=[OptimizeAll()],
            ),
            # Re-tune the latent
            TrainerPhase(
                lr=1.0e-4,
                max_itr=10,
                freq_valid=10,
                patience=5,
                quantize_model=False,
                quantizer_type=STEType(),
                quantizer_noise_type=NoQuantizationNoiseType(),
                optimized_module=[OptimizeLatent()],  # ! Only fine tune the latent
                softround_temperature=(1e-4, 1e-4),
                noise_parameter=(
                    1.0,
                    1.0,
                ),  # not used since quantizer type is "ste"
            ),
        ]

        self.warmup = Warmup(
            [
                WarmupPhase(candidates=3, training_phase=TrainerPhase(max_itr=10)),
                WarmupPhase(candidates=2, training_phase=TrainerPhase(max_itr=10)),
            ]
        )


class PresetSpeedTest(Preset):
    def __init__(self):
        super().__init__(preset_name="fnlic")
        # 1st stage: with soft round and kumaraswamy noise
        self.training_phases: List[TrainerPhase] = [
            TrainerPhase(
                lr=1e-2,
                max_itr=2000,
                freq_valid=100,
                patience=10000,
                quantize_model=False,
                schedule_lr=True,
                softround_temperature=(0.3, 0.1),
                noise_parameter=(0.25, 0.1),
                quantizer_noise_type=KumaraswamyType(),
                quantizer_type=SoftroundType(),
                optimized_module=[OptimizeAll()],
            )
        ]

        # 2nd stage with STE
        lr = 0.00001
        while lr > 10.0e-6:
            self.training_phases.append(
                TrainerPhase(
                    lr=lr,
                    max_itr=100,
                    freq_valid=10,
                    patience=50,
                    quantize_model=False,
                    schedule_lr=True,
                    # This is only used to parameterize the backward of the quantization
                    softround_temperature=(1e-4, 1e-4),
                    noise_parameter=(
                        1.0,
                        1.0,
                    ),  # Kumaraswamy noise with parameter = 1 --> Uniform noise
                    quantizer_noise_type=NoQuantizationNoiseType(),
                    quantizer_type=STEType(),
                    optimized_module=[OptimizeAll()],
                )
            )
            lr *= 0.8

        # 3rd stage: quantize the networks and then re-tune the latent
        lr = 10.0e-6
        self.training_phases.append(
            TrainerPhase(
                lr=lr,
                max_itr=100,
                freq_valid=100,
                patience=100,
                quantize_model=True,
                schedule_lr=False,
                softround_temperature=(1e-4, 1e-4),
                noise_parameter=(1.0, 1.0),
                quantizer_noise_type=NoQuantizationNoiseType(),
                quantizer_type=STEType(),
                optimized_module=[OptimizeAll()],
            )
        )
        self.training_phases.append(
            TrainerPhase(
                lr=1e-4,
                max_itr=200,
                freq_valid=10,
                patience=50,
                quantize_model=False,
                schedule_lr=False,
                softround_temperature=(1e-4, 1e-4),
                noise_parameter=(1.0, 1.0),
                quantizer_noise_type=NoQuantizationNoiseType(),
                quantizer_type=STEType(),
                optimized_module=[OptimizeLatent()],  # ! Only fine tune the latent
            )
        )

        # 5 candidates, then 2 then 1
        self.warmup = Warmup(
            [
                WarmupPhase(
                    candidates=5,
                    training_phase=TrainerPhase(
                        lr=1e-2,
                        max_itr=200,
                        freq_valid=100,
                        patience=100000,
                        quantize_model=False,
                        schedule_lr=False,
                        softround_temperature=(0.3, 0.3),
                        noise_parameter=(1.0, 0.5),
                        quantizer_noise_type=KumaraswamyType(),
                        quantizer_type=SoftroundType(),
                        optimized_module=[OptimizeAll()],
                    ),
                ),
                WarmupPhase(
                    candidates=2,
                    training_phase=TrainerPhase(
                        lr=1e-2,
                        max_itr=400,
                        freq_valid=100,
                        patience=100000,
                        quantize_model=False,
                        schedule_lr=False,
                        softround_temperature=(0.3, 0.3),
                        noise_parameter=(0.5, 0.25),
                        quantizer_noise_type=KumaraswamyType(),
                        quantizer_type=SoftroundType(),
                        optimized_module=[OptimizeAll()],
                    ),
                ),
            ]
        )


class PresetMeasureSpeed(Preset):
    def __init__(
        self,
        start_lr: float = 1e-2,
        itr_main_training: int = 100,
    ):
        super().__init__(preset_name="measure_speed")

        # Single stage model with the shortest warm-up ever!
        self.training_phases: List[TrainerPhase] = [
            TrainerPhase(
                lr=start_lr,
                max_itr=itr_main_training,
                patience=5000,
                optimized_module=[OptimizeAll()],
                schedule_lr=True,
                quantizer_type=SoftroundType(),
                quantizer_noise_type=KumaraswamyType(),
                softround_temperature=(0.3, 0.1),
                noise_parameter=(0.25, 0.1),
                quantize_model=True,  # ! This is an important parameter
            ),
        ]

        self.warmup = Warmup(
            [
                WarmupPhase(
                    candidates=1,
                    training_phase=TrainerPhase(
                        lr=start_lr,
                        max_itr=1,
                        freq_valid=1,
                        patience=100000,
                        quantize_model=False,
                        schedule_lr=False,
                        softround_temperature=(0.3, 0.3),
                        noise_parameter=(2.0, 2.0),
                        quantizer_noise_type=KumaraswamyType(),
                        quantizer_type=SoftroundType(),
                        optimized_module=[OptimizeAll()],
                    ),
                )
            ]
        )


AVAILABLE_PRESETS: Dict[str, Preset] = {
    # "c3x_intra": PresetC3xIntra,
    # "c3x_inter": PresetC3xInter,
    "debug": PresetDebug,
    "measure_speed": PresetMeasureSpeed,
    "fnlic": PresetFNLIC,  # type: ignore
    "speed_test": PresetSpeedTest,
}
