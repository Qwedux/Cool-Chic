from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from domain.component.core.arm_common import (ContextSize, HiddenLayerSize,
                                              NumberOfHiddenLayers)
from domain.component.core.types.parameters_per_channel import \
    ParametersPerChannel
from torch import Tensor


@dataclass(frozen=True, kw_only=True)
class ValidatedMultiARMConfiguration:
    routing_grid: Tensor
    expert_indices: Sequence[Tensor]

@dataclass(frozen=True, kw_only=True)
class ImageARMParameter:
    context_size: ContextSize
    number_of_hidden_layers: NumberOfHiddenLayers
    hidden_layer_size: HiddenLayerSize
    synthesis_out_params_per_channel: Sequence[ParametersPerChannel]
    use_color_regression: bool
    multi_region_image_arm_specification: ValidatedMultiARMConfiguration