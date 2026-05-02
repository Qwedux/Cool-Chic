# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md
from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class DescriptorNN:
    weight: int | float | str | Tensor | None = None
    bias: int | float | str | Tensor | None = None

    def __getitem__(self, item: str) -> int | float | str | Tensor | None:
        if item == "weight":
            return self.weight
        elif item == "bias":
            return self.bias
        return None

    def __setitem__(self, key: str, value: int | float | str | Tensor) -> None:
        if key == "weight":
            self.weight = value
        elif key == "bias":
            self.bias = value


@dataclass
class DescriptorCoolChic:
    """Contains information about the different sub-networks of Cool-chic."""

    arm: DescriptorNN | None = None
    image_arm: DescriptorNN | None = None
    upsampling: DescriptorNN | None = None
    synthesis: DescriptorNN | None = None

@dataclass(frozen=True)
class HorizontalSplit:
    pass

@dataclass(frozen=True)
class VerticalSplit:
    pass

IMARM_SPLIT_DIRECTION = HorizontalSplit | VerticalSplit
