from __future__ import annotations

from dataclasses import dataclass

from domain.component.core.arm_common import (ContextSize, HiddenLayerSize,
                                              NumberOfHiddenLayers)


@dataclass(frozen=True, kw_only=True)
class LatentArmParameter:
    context_size: ContextSize
    number_of_hidden_layers: NumberOfHiddenLayers
    hidden_layer_size: HiddenLayerSize