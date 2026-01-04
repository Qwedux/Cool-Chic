import struct

import torch
from _io import BufferedReader
from lossless.component.coolchic import CoolChicEncoder
from lossless.io.encoding_interfaces.base_interface import \
    EncodeDecodeInterface


class ImageEncodeDecodeInterface(EncodeDecodeInterface):
    def __init__(self, data: tuple, model: CoolChicEncoder) -> None:
        # raw image data is a tensor of shape [1, C, H, W]
        self.raw_image_data = data[0]
        # raw synth out is a tensor of shape [1, S, H, W]
        self.raw_synth_out = data[1]
        self.model: CoolChicEncoder = model
        self.current_spatial_pos = [0, 0, 0, 0]
        self.header_size = 12  # H, W, C

    def reset_iterators(self) -> None:
        self.current_spatial_pos = [0, 0, 0, 0]

    def _iterator_to_flat_index(self) -> int:
        flat_index = 0
        nested_multiplier = 1
        for dim in range(len(self.current_spatial_pos) - 1, -1, -1):
            flat_index += self.current_spatial_pos[dim] * nested_multiplier
            nested_multiplier *= self.raw_image_data.shape[dim]
        return flat_index

    def _flat_index_to_iterator(self, flat_index: int) -> list[int]:
        res = [0, 0, 0, 0]
        for dim in range(len(self.current_spatial_pos) - 1, -1, -1):
            res[dim] = flat_index % self.raw_image_data.shape[dim]
            flat_index = flat_index // self.raw_image_data.shape[dim]
        return res

    def advance_iterators(self, testing_stop=-1) -> None:
        # FIXME: The order is all wrong here -> we should be encoding pixel by pixel channel by channel,
        # not as current channel first then spatially.
        flat_index = self._iterator_to_flat_index()
        flat_index += 1
        if testing_stop > 0 and flat_index > testing_stop:
            raise StopIteration("Partial stop for testing.")
        if flat_index >= self.raw_image_data.numel():
            raise StopIteration("All pixels have been processed.")
        # compute new spatial pos from flat index
        self.current_spatial_pos = self._flat_index_to_iterator(flat_index)

    def get_next_predictor_features(self) -> torch.Tensor:
        b, c, h, w = self.current_spatial_pos
        im_r = self.raw_image_data[b, 0, :, :]
        im_g = self.raw_image_data[b, 1, :, :]
        im_b = self.raw_image_data[b, 2, :, :]
        context_r = self.model.image_arm.get_neighbor_context(im_r, h=h, w=w).flatten()
        context_g = self.model.image_arm.get_neighbor_context(im_g, h=h, w=w).flatten()
        context_b = self.model.image_arm.get_neighbor_context(im_b, h=h, w=w).flatten()
        context = torch.cat([context_r, context_g, context_b], dim=0)[None, :]
        r_syn_o = self.raw_synth_out[b, :, h, w]
        im_known = self.raw_image_data[b, :c, h, w]
        full_feature = torch.cat([context, r_syn_o[None, :], im_known[None, :]], dim=1)
        return full_feature

    def get_pdf_parameters(self, features: torch.Tensor) -> tuple:
        current_channel = self.get_channel_idx()
        _, _, h, w = self.current_spatial_pos

        cutoffs = [
            sum(self.model.image_arm.params.synthesis_out_params_per_channel[:i])
            for i in range(len(self.model.image_arm.params.synthesis_out_params_per_channel) + 1)
        ]
        start_idx = cutoffs[current_channel] + self.model.image_arm.params.context_size * 3
        end_idx = cutoffs[current_channel + 1] + self.model.image_arm.params.context_size * 3
        relevant_raw_synth_out = features[:, start_idx:end_idx]

        image_arm_out = self.model.image_arm.inference_at_position(
            h, w,
            features,
            relevant_raw_synth_out,
            current_channel,
        )
        # mu, log_scale = torch.chunk(image_arm_out, 2, dim=1)
        mu, scale = self.model.proba_output(image_arm_out, self.raw_image_data)
        return mu[0, 0], scale[0, 0]

    def get_current_element(self):
        b, c, h, w = self.current_spatial_pos
        return self.raw_image_data[b, c, h, w]

    def set_decoded_element(self, element) -> None:
        b, c, h, w = self.current_spatial_pos
        self.raw_image_data[b, c, h, w] = element

    def get_packing_parameters(self) -> bytes:
        b, c, h, w = self.raw_image_data.shape
        return struct.pack("iii", h, w, c)

    def set_packing_parameters(self, bitstream: BufferedReader) -> int:
        header = bitstream.read(self.header_size)
        (h, w, c) = struct.unpack("iii", header)
        # FIXME: currently we ignore the packing parameters for the image
        # as we assume fixed size image known apriori. Technically the parameters are
        # in the stream so bpd measurement is unaffected.
        return self.header_size

    def get_channel_idx(self) -> int:
        return self.current_spatial_pos[1]
