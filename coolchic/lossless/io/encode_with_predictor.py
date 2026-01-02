import constriction
import lossless.util.color_transform as color_transform
import torch
from lossless.io.distributions import calculate_probability_distribution
from lossless.io.encoding_interfaces import EncodeDecodeInterface
from lossless.io.types import POSSIBLE_ENCODING_DISTRIBUTIONS


def encode_with_predictor(
    enc_dec_interface: EncodeDecodeInterface,
    ct: color_transform.ColorBitdepths = color_transform.YCoCgBitdepths(),
    distribution: POSSIBLE_ENCODING_DISTRIBUTIONS = "logistic",
    output_path="./test-workdir/encoder_size_test/coolchic_encoded.binary",
):
    enc_dec_interface.reset_iterators()
    
    enc = constriction.stream.stack.AnsCoder() # type: ignore
    bits_theoretical = 0

    with torch.no_grad():
        symbols_to_encode = []
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

                symbols_to_encode.append(
                    enc_dec_interface.get_current_element().int().item() - ct.ranges_int[ch_ind][0]
                )
                bits_theoretical += -torch.log2(
                    prob_distributions[-1][symbols_to_encode[-1]]
                ).item()
                enc_dec_interface.advance_iterators()
            except StopIteration:
                break

        for symbol_index, symbol_to_encode, prob_distribution in zip(
            list(range(len(symbols_to_encode))),
            reversed(symbols_to_encode),
            reversed(prob_distributions),
        ):
            model = constriction.stream.model.Categorical(  # type: ignore
                prob_distribution.detach().cpu().numpy(), perfect=False
            )
            try:
                enc.encode_reverse(symbol_to_encode, model)
            except Exception as e:
                print(f"probability: {prob_distribution}")
                print(f"Error encoding symbol {symbol_to_encode} at index {symbol_index}")
                raise e

    bitstream = enc.get_compressed()
    bitstream.tofile(output_path)
    with open(output_path, "rb") as f:
        original_data = f.read()
    with open(output_path, "wb") as f:
        # Pack two 32-bit integers into binary
        f.write(enc_dec_interface.get_packing_parameters())
        f.write(original_data)

    print(f"Theoretical bits per sub pixel: {bits_theoretical/len(symbols_to_encode)}")

    return bitstream, symbols_to_encode, prob_distributions
