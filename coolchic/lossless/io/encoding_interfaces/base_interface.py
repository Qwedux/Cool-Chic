import struct

import torch
from _io import BufferedReader
from lossless.component.coolchic import CoolChicEncoder
from lossless.util.color_transform import ColorBitdepths


class EncodeDecodeInterface:
    """This class provides unified interface for encoding and decoding with predictors irrespective of the underying data structure.

    The idea is to abstract the encoding and decoding logic from the specifics of what is being encoded - latents, image pixels, etc.
    """

    def __init__(self, data, model: CoolChicEncoder, ct_range: ColorBitdepths) -> None:
        self.data = data
        self.model = model
        self.ct_range = ct_range

    def reset_iterators(self) -> None:
        raise NotImplementedError

    def advance_iterators(self, testing_stop: int = -1) -> None:
        raise NotImplementedError

    def get_next_predictor_features(self) -> torch.Tensor:
        raise NotImplementedError

    def get_pdf_parameters(self, features: torch.Tensor) -> tuple:
        return self.model(features)

    def get_current_element(self):
        raise NotImplementedError
    
    def get_current_element_int(self) -> int:
        raise NotImplementedError
    
    def set_decoded_element(self, element) -> None:
        raise NotImplementedError

    def get_packing_parameters(self) -> bytes:
        raise NotImplementedError

    def set_packing_parameters(self, bitstream: BufferedReader) -> int:
        raise NotImplementedError

    def get_channel_idx(self) -> int:
        raise NotImplementedError
