import torch
import torchac
from lossless.util.distribution import compute_logistic_cdfs
from typing import Literal
from lossless.util.color_transform import ColorBitdepths

POSSIBLE_DISTRIBUTIONS = Literal["logistic", "laplace"]


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
    distribution: POSSIBLE_DISTRIBUTIONS,
    channel_idx: int,
):
    """Calculate Logistic probability distribution for arithmetic coding.
    Works for any shape of mu and s, adds one dimension at the end for the probability axis.
    """
    # Create the base tensor of quantized values
    new_tensor = torch.linspace(0, 1, steps=256, device=mu.device)  # [256]
    new_shape = (*mu.shape, 256)  # add one dimension at the end
    new_tensor = new_tensor.view(*([1] * mu.ndim), 256).expand(*new_shape)

    # Compute boundaries for each bin
    x_minus = new_tensor - 0.5 / 256
    x_plus = new_tensor + 0.5 / 256

    # Expand mu and s with one trailing dimension
    mu_expanded = mu.unsqueeze(-1)
    s_expanded = s.unsqueeze(-1)

    # Logistic CDF difference between bin edges (use the logistic cdf function)
    if distribution == "laplace":
        cdf_minus = _laplace_cdf(x_minus, mu_expanded, s_expanded)
        cdf_plus = _laplace_cdf(x_plus, mu_expanded, s_expanded)
    elif distribution == "logistic":
        cdf_minus = _logistic_cdf(x_minus, mu_expanded, s_expanded)
        cdf_plus = _logistic_cdf(x_plus, mu_expanded, s_expanded)

    prob_t = cdf_plus - cdf_minus
    prob_t = torch.clamp_min(prob_t, 2 ** (-16))
    prob_t = prob_t / prob_t.sum(dim=-1, keepdim=True)

    return prob_t


def dist_to_cfd(prob_dist: torch.Tensor) -> torch.Tensor:
    # we go from probability table in the last dimension [..., num_symbols] to CDF table in the last dimension [..., num_symbols + 1]
    cdf = torch.zeros(
        *prob_dist.shape[:-1],
        prob_dist.shape[-1] + 1,
        device=prob_dist.device,
        dtype=torch.float32,
    )
    cdf[..., 1:] = prob_dist.cumsum(dim=-1)
    cdf[..., -1] = 0  # ensure last value is exactly 0
    cdf = torch.round(cdf * float(1 << 16)).to(torch.int32)
    cdf[..., -1] = 0  # ensure last value is exactly 2^16
    return cdf.to(torch.int16)


def encode(
    x: torch.Tensor,
    mu: torch.Tensor,
    scale: torch.Tensor,
    color_bitdepths: ColorBitdepths,
    distribution: POSSIBLE_DISTRIBUTIONS = "logistic",
):
    # this undoes normalization
    x_reshape = torch.floor(x * 255).to(torch.int16).cpu()

    byte_strings = []
    for i in range(3):
        symbols = x_reshape[:, i : i + 1, ...]
        # print(
        #     f"Channel {i}: symbols shape {symbols.shape}, min {symbols.min()}, max {symbols.max()}"
        # )
        cur_cdfs = dist_to_cfd(
            calculate_probability_distribution(
                mu[:, i : i + 1, ...],
                scale[:, i : i + 1, ...],
                color_bitdepths=color_bitdepths,
                distribution=distribution,
                channel_idx=i,
            )
        ).cpu()
        byte_strings.append(
            torchac.encode_int16_normalized_cdf(cur_cdfs, symbols)
        )
    return byte_strings


def decode(
    byte_strings: list,
    mu: torch.Tensor,
    scale: torch.Tensor,
    color_bitdepths: ColorBitdepths,
    distribution: POSSIBLE_DISTRIBUTIONS = "logistic",
):
    assert len(byte_strings) == 3

    _, _, h, w = mu.size()
    x_rec = torch.zeros(1, 3, h, w)

    # Channel 0 (Red)
    cur_cdfs_r = dist_to_cfd(
        calculate_probability_distribution(
            mu[:, :1, ...],
            scale[:, :1, ...],
            color_bitdepths=color_bitdepths,
            distribution=distribution,
            channel_idx=0,
        )
    ).cpu()
    symbols_r = torchac.decode_int16_normalized_cdf(cur_cdfs_r, byte_strings[0])
    print(
        f"Decoded R: shape {symbols_r.shape}, min {symbols_r.min()}, max {symbols_r.max()}"
    )
    x_r = symbols_r.reshape(1, 1, h, w).float() / 255
    x_rec[:, 0:1, ...] = x_r  # FIX: was using 3*i (which is 0), should be 0:1

    # Channel 1 (Green)
    cur_cdfs_g = dist_to_cfd(
        calculate_probability_distribution(
            mu[:, 1:2, ...],
            scale[:, 1:2, ...],
            color_bitdepths=color_bitdepths,
            distribution=distribution,
            channel_idx=1,
        )
    ).cpu()
    symbols_g = torchac.decode_int16_normalized_cdf(cur_cdfs_g, byte_strings[1])
    print(
        f"Decoded G: shape {symbols_g.shape}, min {symbols_g.min()}, max {symbols_g.max()}"
    )
    x_g = symbols_g.reshape(1, 1, h, w).float() / 255
    x_rec[:, 1:2, ...] = x_g  # FIX: was using 3*i+1 (which is 1), should be 1:2

    # Channel 2 (Blue)
    cur_cdfs_b = dist_to_cfd(
        calculate_probability_distribution(
            mu[:, 2:3, ...],
            scale[:, 2:3, ...],
            color_bitdepths=color_bitdepths,
            distribution=distribution,
            channel_idx=2,
        )
    ).cpu()
    symbols_b = torchac.decode_int16_normalized_cdf(cur_cdfs_b, byte_strings[2])
    print(
        f"Decoded B: shape {symbols_b.shape}, min {symbols_b.min()}, max {symbols_b.max()}"
    )
    x_b = symbols_b.reshape(1, 1, h, w).float() / 255
    x_rec[:, 2:3, ...] = x_b  # FIX: was using 3*i+2 (which is 2), should be 2:3

    return x_rec
