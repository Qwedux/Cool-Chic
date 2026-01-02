import struct

import constriction
import lossless.util.color_transform as color_transform
import numpy as np
import torch
from lossless.io.distributions import calculate_probability_distribution
from lossless.io.encoding_interfaces import EncodeDecodeInterface
from lossless.io.types import POSSIBLE_ENCODING_DISTRIBUTIONS


def decode_quick_check(
    bitstream_path: str,
    prob_distributions: list[torch.Tensor],
    ct: color_transform.ColorBitdepths = color_transform.YCoCgBitdepths(),
    offset: int = 4,
):
    with open(bitstream_path, "rb") as f:
        header = f.read(offset)  # 3 integers * 4 bytes each
        num_symbols = struct.unpack("i" * (offset // 4), header)
    bitstream = np.fromfile(bitstream_path, dtype=np.uint32, offset=offset)
    dec = constriction.stream.stack.AnsCoder(bitstream)  # type: ignore

    decoded_symbols = []
    with torch.no_grad():
        for prob_distribution in prob_distributions:
            prob_array = prob_distribution.detach().cpu().flatten().numpy()
            model = constriction.stream.model.Categorical(prob_array, perfect=False)  # type: ignore
            decoded_char = torch.tensor(dec.decode(model, 1)[0]) + ct.ranges_int[0][0]
            decoded_symbols.append(decoded_char.item())
    return decoded_symbols


def decode_with_predictor(
    bitstream_path: str,
    enc_dec_interface: EncodeDecodeInterface,
    ct=color_transform.ColorBitdepths(),
    distribution: POSSIBLE_ENCODING_DISTRIBUTIONS = "logistic",
):
    enc_dec_interface.reset_iterators()
    with open(bitstream_path, "rb") as f:
        offset = enc_dec_interface.set_packing_parameters(f)
    print(f"offset: {offset}")
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
                        mu - ct.ranges_int[ch_ind][0] / ct.scaling_factors[ch_ind],
                        scale,
                        color_bitdepths=ct,
                        distribution=distribution,
                        channel_idx=ch_ind,
                    )
                )
                model = constriction.stream.model.Categorical(  # type: ignore
                    prob_distributions[-1].detach().cpu().numpy(), perfect=False
                )
                symbols_decoded.append(
                    (torch.tensor(dec.decode(model, 1)[0]) + ct.ranges_int[ch_ind][0]).item()
                )
                enc_dec_interface.set_decoded_element(symbols_decoded[-1])
                enc_dec_interface.advance_iterators()
            except StopIteration:
                break
    return symbols_decoded, prob_distributions
