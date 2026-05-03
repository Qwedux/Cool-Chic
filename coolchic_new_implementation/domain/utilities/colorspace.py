from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class YCoCg:
    pass

@dataclass(frozen=True)
class RGB:
    pass

SupportedColorspace = YCoCg | RGB