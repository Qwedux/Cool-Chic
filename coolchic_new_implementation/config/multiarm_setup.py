from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

NumPartsPerCol = NewType("NumPartsPerCol", int)
NumPartsPerRow = NewType("NumPartsPerRow", int)

@dataclass(frozen=True, kw_only=True)
class GridSplit:
    num_parts_per_col: NumPartsPerCol
    num_parts_per_row: NumPartsPerRow

@dataclass(frozen=True)
class QuadtreeSplit:
    pass

MultiArmSplitSpecification = GridSplit | QuadtreeSplit
