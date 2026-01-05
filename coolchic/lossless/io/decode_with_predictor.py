import struct

import constriction
import lossless.util.color_transform as color_transform
import numpy as np
import torch
from lossless.io.distributions import calculate_probability_distribution
from lossless.io.encoding_interfaces.base_interface import \
    EncodeDecodeInterface
from lossless.io.types import POSSIBLE_ENCODING_DISTRIBUTIONS


def decode_quick_check(
    prob_distributions: list[torch.Tensor],
    channel_indices: list[int],
    bitstream,
    bitstream_path: str | None= "./test-workdir/encoder_size_test/coolchic_encoded_image.binary",
    ct: color_transform.ColorBitdepths = color_transform.YCoCgBitdepths(),
    offset: int = 4,
):
    if bitstream_path is not None:
        with open(bitstream_path, "rb") as f:
            header = f.read(offset)  # 3 integers * 4 bytes each
            num_symbols = struct.unpack("i" * (offset // 4), header)
        bitstream = np.fromfile(bitstream_path, dtype=np.uint32, offset=offset)
    
    dec = constriction.stream.stack.AnsCoder(bitstream)  # type: ignore

    decoded_symbols = []
    with torch.no_grad():
        for i, prob_distribution in enumerate(prob_distributions):
            prob_array = prob_distribution.detach().cpu().flatten().numpy()
            model = constriction.stream.model.Categorical(prob_array, perfect=False)  # type: ignore
            decoded_char = (
                torch.tensor(dec.decode(model, 1)[0]) + ct.ranges_int[channel_indices[i]][0]
            )
            decoded_symbols.append(decoded_char.item())
    return decoded_symbols


def decode_with_predictor(
    enc_dec_interface: EncodeDecodeInterface,
    bitstream,
    bitstream_path: str | None= "./test-workdir/encoder_size_test/coolchic_encoded_image.binary",
    # bitstream_path: str = "./test-workdir/encoder_size_test/coolchic_encoded_image.binary",
    distribution: POSSIBLE_ENCODING_DISTRIBUTIONS = "logistic",
):
    enc_dec_interface.reset_iterators()
    if bitstream_path is not None:
        with open(bitstream_path, "rb") as f:
            offset = enc_dec_interface.set_packing_parameters(f)
        bitstream = np.fromfile(bitstream_path, dtype=np.uint32, offset=offset)
    dec = constriction.stream.stack.AnsCoder(bitstream)  # type: ignore

    with torch.no_grad():
        symbols_decoded = []
        prob_distributions = []
        while True:
            try:
                features = enc_dec_interface.get_next_predictor_features()
                mu, scale = enc_dec_interface.get_pdf_parameters(features)
                ch_ind = enc_dec_interface.get_channel_idx()
                prob_distributions.append(
                    calculate_probability_distribution(
                        mu
                        - enc_dec_interface.ct_range.ranges_int[ch_ind][0]
                        / enc_dec_interface.ct_range.scaling_factors[ch_ind],
                        scale,
                        color_bitdepths=enc_dec_interface.ct_range,
                        distribution=distribution,
                        channel_idx=ch_ind,
                    )
                )
                model = constriction.stream.model.Categorical(  # type: ignore
                    prob_distributions[-1].detach().cpu().numpy(), perfect=False
                )
                symbols_decoded.append(
                    (
                        torch.tensor(dec.decode(model, 1)[0])
                        + enc_dec_interface.ct_range.ranges_int[ch_ind][0]
                    ).item()
                )
                enc_dec_interface.set_decoded_element(
                    symbols_decoded[-1] / enc_dec_interface.ct_range.scaling_factors[ch_ind]
                )
                enc_dec_interface.advance_iterators()
            except StopIteration:
                break
    return symbols_decoded, prob_distributions
