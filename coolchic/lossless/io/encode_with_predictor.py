import constriction
import torch
from lossless.io.distributions import calculate_probability_distribution
from lossless.io.encoding_interfaces.base_interface import \
    EncodeDecodeInterface
from lossless.io.types import POSSIBLE_ENCODING_DISTRIBUTIONS
from lossless.util.logger import TrainingLogger


def encode_with_predictor(
    enc_dec_interface: EncodeDecodeInterface,
    logger: TrainingLogger | None,
    distribution: POSSIBLE_ENCODING_DISTRIBUTIONS = "logistic",
    output_path: str | None = "./test-workdir/encoder_size_test/coolchic_encoded_image.binary",
):
    enc_dec_interface.reset_iterators()

    enc = constriction.stream.stack.AnsCoder()  # type: ignore
    bits_theoretical = 0

    with torch.no_grad():
        symbols_to_encode = []
        prob_distributions = []
        raw_values_of_symbols_to_encode = []
        channel_indices = []
        while True:
            try:
                features = enc_dec_interface.get_next_predictor_features()
                mu, scale = enc_dec_interface.get_pdf_parameters(features)

                ch_ind = enc_dec_interface.get_channel_idx()
                channel_indices.append(ch_ind)
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
                raw_values_of_symbols_to_encode.append(
                    (
                        enc_dec_interface.get_current_element()
                        * enc_dec_interface.ct_range.scaling_factors[ch_ind]
                    )
                    .int()
                    .item()
                )
                symbols_to_encode.append(
                    raw_values_of_symbols_to_encode[-1]
                    - enc_dec_interface.ct_range.ranges_int[ch_ind][0]
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
                exception_string = (
                    f"probability: {prob_distribution}\n"
                    + f"Error encoding symbol {symbol_to_encode} at index {symbol_index}"
                )
                if logger is not None:
                    logger.log_result(exception_string)
                else:
                    print(exception_string)
                raise e

    bitstream = enc.get_compressed()
    if output_path is not None:
        bitstream.tofile(output_path)
        with open(output_path, "rb") as f:
            original_data = f.read()
        with open(output_path, "wb") as f:
            f.write(enc_dec_interface.get_packing_parameters())
            f.write(original_data)

    theoretical_bpd_string = f"Theoretical bits per sub pixel: {bits_theoretical/len(symbols_to_encode)*enc_dec_interface.normalization_constant}"
    if logger is not None:
        logger.log_result(theoretical_bpd_string)
    else:
        print(theoretical_bpd_string)
    return bitstream, raw_values_of_symbols_to_encode, prob_distributions, channel_indices
