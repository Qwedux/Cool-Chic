import struct

import torch
from _io import BufferedReader
from lossless.component.coolchic import CoolChicEncoder


class EncodeDecodeInterface:
    """This class provides unified interface for encoding and decoding with predictors irrespective of the underying data structure.

    The idea is to abstract the encoding and decoding logic from the specifics of what is being encoded - latents, image pixels, etc.
    """

    def __init__(self, data, model: CoolChicEncoder) -> None:
        self.data = data
        self.predictor = model

    def reset_iterators(self) -> None:
        raise NotImplementedError

    def advance_iterators(self) -> None:
        raise NotImplementedError

    def get_next_predictor_features(self) -> torch.Tensor:
        raise NotImplementedError

    def get_pdf_parameters(self, features: torch.Tensor) -> tuple:
        return self.predictor(features)

    def get_current_element(self):
        raise NotImplementedError

    def set_decoded_element(self, element) -> None:
        raise NotImplementedError

    def get_packing_parameters(self) -> bytes:
        raise NotImplementedError

    def set_packing_parameters(self, bitstream: BufferedReader) -> int:
        raise NotImplementedError

    def get_channel_idx(self) -> int:
        raise NotImplementedError