from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class YCoCg:
    pass

@dataclass(frozen=True, slots=True)
class RGB:
    pass

PossibleColorspace = YCoCg | RGB