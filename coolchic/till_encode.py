import torch
import constriction
import struct
from lossless.component.types import POSSIBLE_ENCODING_DISTRIBUTIONS
import lossless.util.color_transform as color_transform
from lossless.util.color_transform import ColorBitdepths
import numpy as np


def get_bits_per_pixel(w, h, c, encoded_bytes):
    num_pixels = w * h * c
    num_bits = 0
    for bytes_channel in encoded_bytes:
        num_bits += len(bytes_channel) * 8
    return num_bits / num_pixels


def _laplace_cdf(
    x: torch.Tensor, expectation: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Compute the laplace cumulative evaluated in x. All parameters
    must have the same dimension.
    Re-implemented here coz it is faster than calling the Laplace distribution
    from torch.distributions.

    Args:
        x (Tensor): Where the cumulative if evaluated.
        expectation (Tensor): Expectation.
        scale (Tensor): Scale

    Returns:
        Tensor: CDF(x, mu, scale)
    """
    shifted_x = x - expectation
    return 0.5 - 0.5 * (shifted_x).sign() * torch.expm1(
        -(shifted_x).abs() / scale
    )


def _logistic_cdf(
    x: torch.Tensor, mu: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """Compute the logistic cumulative evaluated in x. All parameters
    must have the same dimension.
    Re-implemented here coz it is faster than calling the Logistic distribution
    from torch.distributions.

    Args:
        x (Tensor): Where the cumulative if evaluated.
        mu (Tensor): Expectation.
        s (Tensor): Scale

    Returns:
        Tensor: CDF(x, mu, scale)
    """
    z = (x - mu) / s
    return torch.sigmoid(z)


def calculate_probability_distribution(
    mu,
    s,
    color_bitdepths: ColorBitdepths,
    distribution: POSSIBLE_ENCODING_DISTRIBUTIONS,
    channel_idx: int,
):
    """Calculate Logistic probability distribution for arithmetic coding.
    Works for any shape of mu and s, adds one dimension at the end for the probability axis.
    """
    # Create the base tensor of quantized values
    new_tensor = torch.linspace(
        0.0,
        (
            color_bitdepths.ranges_int[channel_idx][1]
            - color_bitdepths.ranges_int[channel_idx][0]
        )
        / color_bitdepths.scaling_factors[channel_idx],
        steps=color_bitdepths.bins[channel_idx],
        device=mu.device,
    )
    new_shape = (
        *mu.shape,
        color_bitdepths.bins[channel_idx],
    )  # add one dimension at the end
    new_tensor = new_tensor.view(
        *([1] * mu.ndim), color_bitdepths.bins[channel_idx]
    ).expand(*new_shape)

    # Compute boundaries for each bin
    x_minus = new_tensor - 0.5 / color_bitdepths.bins[channel_idx]
    x_plus = new_tensor + 0.5 / color_bitdepths.bins[channel_idx]

    # Expand mu and s with one trailing dimension
    mu_expanded = mu.unsqueeze(-1)
    s_expanded = s.unsqueeze(-1)

    # Logistic CDF difference between bin edges (use the logistic cdf function)
    if distribution == "laplace":
        cdf_minus = _laplace_cdf(x_minus, mu_expanded, s_expanded)
        cdf_plus = _laplace_cdf(x_plus, mu_expanded, s_expanded)
    else:
        cdf_minus = _logistic_cdf(x_minus, mu_expanded, s_expanded)
        cdf_plus = _logistic_cdf(x_plus, mu_expanded, s_expanded)
    prob_t = cdf_plus - cdf_minus
    if distribution == "dummy":
        prob_t = torch.ones_like(cdf_plus)
    prob_t = torch.clamp_min(prob_t, 2 ** (-16))
    prob_t = prob_t / prob_t.sum(dim=-1, keepdim=True)

    assert torch.all(
        torch.isclose(prob_t.sum(dim=-1), torch.ones_like(mu))
    ), "Probabilities do not sum to 1"
    assert torch.all(prob_t.min() > 0), "Some probabilities are zero"
    return prob_t


import torch
import constriction
import struct


def encode(
    x: torch.Tensor,
    mu: torch.Tensor,
    scale: torch.Tensor,
    ct: color_transform.ColorBitdepths = color_transform.YCoCgBitdepths(),
    distribution: POSSIBLE_ENCODING_DISTRIBUTIONS = "logistic",
    output_path="./test-workdir/encoder_size_test/coolchic_encoded.binary",
):
    x = x * 255.0
    x_reshape = x.to(torch.int16).cpu()

    B, C, H, W = x.shape
    enc = constriction.stream.stack.AnsCoder()  # type: ignore
    bits_theoretical = 0

    with torch.no_grad():
        scale_theoretical_bits = 0
        mu = mu.flatten(2, 3).permute(0, 2, 1)  # B, H*W, C
        scale = scale.flatten(2, 3).permute(0, 2, 1)  # B, H*W, C

        probs_logistic = [
            calculate_probability_distribution(
                mu - ct.ranges_int[ch_ind][0] / ct.scaling_factors[ch_ind],
                scale,
                color_bitdepths=ct,
                distribution=distribution,
                channel_idx=ch_ind,
            )
            for ch_ind in range(C)
        ]

        scale_theoretical_bits = 0
        for wh in range(x.shape[2] * x.shape[3]):
            for c in range(x.shape[1]):
                sym: int = (
                    x_reshape.flatten(2, 3).permute(0, 2, 1)[0, -wh - 1, -c - 1]
                ).int().item() - ct.ranges_int[-c - 1][0]
                prob_t = probs_logistic[-c - 1][0, -wh - 1, -c - 1]
                scale_theoretical_bits += -torch.log2(prob_t[sym]).item()
                model = constriction.stream.model.Categorical(  # type: ignore
                    prob_t.detach().cpu().numpy(), perfect=False
                )
                try:
                    enc.encode_reverse(sym, model)
                except Exception as e:

                    print(f"Error encoding symbol {sym} for channel {c}: {e}")
                    raise e
        bits_theoretical += scale_theoretical_bits

    bitstream = enc.get_compressed()
    bitstream.tofile(output_path)
    with open(output_path, "rb") as f:
        original_data = f.read()
    with open(output_path, "wb") as f:
        # Pack two 32-bit integers into binary
        f.write(struct.pack("iii", H, W, C))
        f.write(original_data)

    print(f"Theoretical bits per sub pixel: {bits_theoretical/float(W*H*C)}")

    return bitstream, probs_logistic


def decode(
    bitstream_path,
    mu: torch.Tensor,
    scale: torch.Tensor,
    ct=color_transform.ColorBitdepths(),
    distribution: POSSIBLE_ENCODING_DISTRIBUTIONS = "logistic",
):
    with open(bitstream_path, "rb") as f:
        header = f.read(12)  # 3 integers * 4 bytes each
        H, W, C = struct.unpack("iii", header)
    bitstream = np.fromfile(bitstream_path, dtype=np.uint32, offset=12)
    dec = constriction.stream.stack.AnsCoder(bitstream)  # type: ignore

    x = -torch.ones(1, C, H, W)
    with torch.no_grad():
        mu = mu.flatten(2, 3).permute(0, 2, 1)  # B, H*W, C
        scale = scale.flatten(2, 3).permute(0, 2, 1)  # B, H*W, C

        probs_logistic = [
            calculate_probability_distribution(
                mu - ct.ranges_int[ch_ind][0] / ct.scaling_factors[ch_ind],
                scale,
                color_bitdepths=ct,
                distribution=distribution,
                channel_idx=ch_ind,
            )
            for ch_ind in range(C)
        ]
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    prob = probs_logistic[c][0, h * W + w, c]
                    prob_array = prob.detach().cpu().flatten().numpy()
                    model = constriction.stream.model.Categorical(  # type: ignore
                        prob_array, perfect=False
                    )
                    decoded_char = (
                        torch.tensor(dec.decode(model, 1)[0]).float()
                        + ct.ranges_int[c][0]
                    )
                    x[0, c, h, w] = decoded_char

    x = x.cpu() / 255

    return x, probs_logistic
