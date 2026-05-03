from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    pass

@dataclass(frozen=True)
class PresetEncode(Preset):
    pass

@dataclass(frozen=True)
class PresetMeasureSpeed(Preset):
    pass

@dataclass(frozen=True)
class PresetDebug(Preset):
    pass

@dataclass(frozen=True)
class PresetWarmupTest:
    pass

@dataclass(frozen=True)
class PresetPretrainedModelFineTuning:
    pass


AvailablePresets = PresetEncode | PresetMeasureSpeed | PresetDebug | PresetWarmupTest | PresetPretrainedModelFineTuning