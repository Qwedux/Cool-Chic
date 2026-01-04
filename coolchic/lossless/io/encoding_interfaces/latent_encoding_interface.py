import struct

import torch
from _io import BufferedReader
from lossless.component.coolchic import CoolChicEncoder
from lossless.io.encoding_interfaces.base_interface import \
    EncodeDecodeInterface


class LatentEncodeDecodeInterface(EncodeDecodeInterface):
    def __init__(self, data: list[torch.Tensor], model: CoolChicEncoder) -> None:
        # latents are a list of tensors of shape [1, C, H, W]
        super().__init__(data, model)
        self.model = model
        self.current_latent_idx = 0
        self.current_spatial_pos = [0, 0, 0, 0]
        self.header_size = 4

    def reset_iterators(self) -> None:
        self.current_latent_idx = 0
        self.current_spatial_pos = [0, 0, 0, 0]

    def _iterator_to_flat_index(self) -> int:
        latent_shape = self.data[self.current_latent_idx].shape
        flat_index = 0
        nested_multiplier = 1
        for index in range(len(self.current_spatial_pos) - 1, -1, -1):
            flat_index += self.current_spatial_pos[index] * nested_multiplier
            nested_multiplier *= latent_shape[index]
        return flat_index

    def _flat_index_to_iterator(self, flat_index: int) -> list[int]:
        latent_shape = self.data[self.current_latent_idx].shape
        res = [0, 0, 0, 0]
        for dim in range(len(self.current_spatial_pos) - 1, -1, -1):
            res[dim] = flat_index % latent_shape[dim]
            flat_index = flat_index // latent_shape[dim]
        return res

    def advance_iterators(self) -> None:
        if self.current_latent_idx >= len(self.data):
            raise StopIteration("All latents have been processed.")

        flat_index = self._iterator_to_flat_index()
        flat_index += 1

        if flat_index >= self.data[self.current_latent_idx].numel():
            # move to next latent
            self.current_latent_idx += 1
            if self.current_latent_idx >= len(self.data):
                raise StopIteration("All latents have been processed.")
            self.current_spatial_pos = [0, 0, 0, 0]
            return
        # compute new spatial pos from flat index
        self.current_spatial_pos = self._flat_index_to_iterator(flat_index)

    def get_next_predictor_features(self) -> torch.Tensor:
        b, c, h, w = self.current_spatial_pos
        latent = self.data[self.current_latent_idx][b, c, :, :]
        context = self.model.arm.get_neighbor_context(latent, h=h, w=w)
        return context

    def get_pdf_parameters(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # I assume that the features have shape [1, context_size]
        # print("Getting PDF parameters for latent index", self.current_latent_idx, "at position", self.current_spatial_pos)
        # print("Features:", features.dtype, features.shape, features)
        # raise StopExecution()
        mu, scale, log_scale = super().get_pdf_parameters(features)
        return mu[0], scale[0]

    def get_current_element(self):
        b, c, h, w = self.current_spatial_pos
        return self.data[self.current_latent_idx][b, c, h, w]

    def set_decoded_element(self, element) -> None:
        b, c, h, w = self.current_spatial_pos
        self.data[self.current_latent_idx][b, c, h, w] = element

    def get_packing_parameters(self) -> bytes:
        # this could
        total_elems = sum([latent.numel() for latent in self.data])
        return struct.pack("i", total_elems)

    def set_packing_parameters(self, bitstream: BufferedReader) -> int:
        header = bitstream.read(self.header_size)
        (total_elems,) = struct.unpack("i", header)
        # FIXME: currently we ignore the packing parameters for latents
        # as we assume fixed size latents known from the model
        return self.header_size

    def get_channel_idx(self) -> int:
        return self.current_spatial_pos[1]


