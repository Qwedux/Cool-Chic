import struct

import torch
from _io import BufferedReader


class EncodeDecodeInterface:
    """This class provides unified interface for encoding and decoding with predictors irrespective of the underying data structure.

    The idea is to abstract the encoding and decoding logic from the specifics of what is being encoded - latents, image pixels, etc.
    """

    def __init__(self, data, predictor) -> None:
        self.data = data
        self.predictor = predictor

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


class LatentEncodeDecodeInterface(EncodeDecodeInterface):
    def __init__(self, data: list[torch.Tensor], predictor: arm.Arm) -> None:
        # latents are a list of tensors of shape [1, C, H, W]
        super().__init__(data, predictor)
        self.predictor = predictor
        self.current_latent_idx = 0
        self.current_spatial_pos = [0, 0, 0, 0]
        self.header_size = 4
        self.testing_stop = -1  # FIXME: temporary stop for testing

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
        # FIXME: temporary stop for testing
        if flat_index > self.testing_stop and self.testing_stop > 0:
            raise StopIteration("Partial stop for testing.")

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
        context = self.predictor.get_neighbor_context(latent, h=h, w=w)
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


class ImageEncodeDecodeInterface(EncodeDecodeInterface):
    def __init__(self, data: tuple, predictor: arm_image.ImageArm) -> None:
        # raw image data is a tensor of shape [1, C, H, W]
        self.raw_image_data = data[0]
        # raw synth out is a tensor of shape [1, 9, H, W]
        self.raw_synth_out = data[1]
        self.predictor = predictor
        self.predictor = predictor
        self.current_spatial_pos = [0, 0, 0, 0]
        self.header_size = 12  # H, W, C
        self.cutoffs = [
            sum(predictor.synthesis_out_params_per_channel[:i])
            for i in range(len(predictor.synthesis_out_params_per_channel) + 1)
        ]
        self.testing_stop = -1  # FIXME: temporary stop for testing

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

    def advance_iterators(self) -> None:
        # FIXME: The order is all wrong here -> we should be encoding pixel by pixel channel by channel,
        # not as current channel first then spatially.
        flat_index = self._iterator_to_flat_index()
        flat_index += 1
        if self.testing_stop > 0 and flat_index > self.testing_stop:
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
        context_r = self.predictor.get_neighbor_context(im_r, h=h, w=w).flatten()
        context_g = self.predictor.get_neighbor_context(im_g, h=h, w=w).flatten()
        context_b = self.predictor.get_neighbor_context(im_b, h=h, w=w).flatten()
        context = torch.cat([context_r, context_g, context_b], dim=0)[None, :]
        r_syn_o = self.raw_synth_out[b, :, h, w]
        im_known = self.raw_image_data[b, :c, h, w]
        full_feature = torch.cat([context, r_syn_o[None, :], im_known[None, :]], dim=1)
        return full_feature

    def get_pdf_parameters(self, features: torch.Tensor) -> tuple:
        current_channel = self.get_channel_idx()
        _, _, h, w = self.current_spatial_pos

        start_idx = self.cutoffs[current_channel] + self.predictor.context_size * 3
        end_idx = self.cutoffs[current_channel + 1] + self.predictor.context_size * 3
        # start_idx = current_channel * 2
        # end_idx = start_idx + 2
        relevant_raw_synth_out = features[:, start_idx:end_idx]

        image_arm_out = self.predictor.inference(
            features,
            relevant_raw_synth_out,
            current_channel,
        )
        mu, log_scale = torch.chunk(image_arm_out, 2, dim=1)
        scale = get_scale(log_scale)
        return mu[0,0], scale[0,0]
        # if current_channel == 0:
        #     mu, log_scale = torch.chunk(image_arm_out, 2, dim=1)
        #     scale = get_scale(log_scale)
        #     return mu[0,0], scale[0,0]
        # elif current_channel == 1:
        #     _mu, log_scale, alpha = torch.chunk(image_arm_out, 3, dim=1)
        #     mu = _mu + alpha * self.raw_image_data[0, current_channel - 1, h, w]
        #     scale = get_scale(log_scale)
        #     return mu[0,0], scale[0,0]
        # elif current_channel == 2:
        #     _mu, log_scale, beta, gamma = torch.chunk(image_arm_out, 4, dim=1)
        #     mu = (
        #         _mu
        #         + beta * self.raw_image_data[0, current_channel - 2, h, w]
        #         + gamma * self.raw_image_data[0, current_channel - 1, h, w]
        #     )
        #     scale = get_scale(log_scale)
        #     return mu[0,0], scale[0,0]
        # raise StopIteration(f"Cannot process channel index {current_channel}")

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