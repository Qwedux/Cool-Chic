from dataclasses import dataclass


@dataclass(frozen=True)
class SoftroundAloneType:
    pass

@dataclass(frozen=True)
class SoftroundType:
    pass

@dataclass(frozen=True)
class HardroundType:
    pass

@dataclass(frozen=True)
class STEType:
    pass

@dataclass(frozen=True)
class NoQuantizerType:
    pass

POSSIBLE_QUANTIZER_TYPE = SoftroundAloneType | SoftroundType | HardroundType | STEType | NoQuantizerType
