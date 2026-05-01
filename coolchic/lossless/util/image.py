from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

ImageWidth: TypeAlias = int
ImageHeight: TypeAlias = int


@dataclass(frozen=True, kw_only=True)
class ImageSize:
    width: ImageWidth
    height: ImageHeight
