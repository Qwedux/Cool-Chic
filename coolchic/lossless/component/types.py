# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

from dataclasses import dataclass
from typing import Literal, Optional, Union
from torch import Tensor


from dataclasses import dataclass
from typing import Literal, Optional, Union
from torch import Tensor

@dataclass
class DescriptorNN:
    """Contains information (scale, weight, quantization step, ...) about the
    weights and biases of a neural network."""

    weight: Optional[Union[int, float, str, Tensor]] = None
    bias: Optional[Union[int, float, str, Tensor]] = None

    # def __init__(
    #     self, weight: Optional[Tensor] = None, bias: Optional[Tensor] = None
    # ):
    #     self.weight = weight
    #     self.bias = bias

    def __getitem__(self, item: str) -> Optional[Union[int, float, str, Tensor]]:
        if item == "weight":
            return self.weight
        elif item == "bias":
            return self.bias
        return None

    def __setitem__(self, key: str, value: Union[int, float, str, Tensor]) -> None:
        if key == "weight":
            self.weight = value
        elif key == "bias":
            self.bias = value


@dataclass
class DescriptorCoolChic:
    """Contains information about the different sub-networks of Cool-chic."""

    arm: Optional[DescriptorNN] = None
    # image_arm: Optional[DescriptorNN] = None
    upsampling: Optional[DescriptorNN] = None
    synthesis: Optional[DescriptorNN] = None


# For now, it is only possible to have a Cool-chic encoder
# with this name i.e. this key in frame_encoder.coolchic_enc
NAME_COOLCHIC_ENC = Literal["lossless"]
