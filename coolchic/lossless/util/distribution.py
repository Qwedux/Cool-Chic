from typing import Tuple

import numpy as np
import torch
from lossless.util.color_transform import ColorBitdepths
from torch import Tensor

rans_freq_precision: int = 16


def get_scale(logscale: Tensor) -> Tensor:
    """
    function to calculate the scale from the log-scale
    """
    return torch.exp(-0.5 * torch.clamp(logscale, min=-10.0, max=13.8155))


def get_mu_scale(params: Tensor) -> Tuple[Tensor, Tensor]:
    """
    function to calculate the mean and scale from the parameters
    """
    mu, logscale = torch.chunk(params, 2, dim=1)
    scale = get_scale(logscale)
    return mu, scale


def modify_prob(prob: Tensor, precision: int) -> Tensor:
    """
    function to modify the frequencies of the symbols that all possible value has at lease 1 frequency
    by adding 1 to the last symbol
    freqs: ndarray (ndist, symbols)
    """
    a = 2 ** (-precision)
    prob = prob.clamp_min(a)
    return prob


def modify_regular_prob(probs: Tensor) -> Tensor:
    """
    function to modify the frequencies of the symbols that all possible value has at lease 1 frequency and the sum of the frequencies is 2**rans_freq_precision
    """
    a = 2 ** (-rans_freq_precision)
    n = 256
    probs = probs * (1 - n * a) + a
    return probs


def compute_logistic_cdfs(
    mu: torch.Tensor, scale: torch.Tensor, bitdepth: int
) -> torch.Tensor:
    """
    function to calculate the cdfs of the Logistic(mu, scale) distribution
    used for encoder
    """
    mu = mu
    scale = scale
    n, c, h, w = mu.shape
    max_v = float((1 << bitdepth) - 1)
    size = 1 << bitdepth
    interval = 1.0 / max_v
    endpoints = torch.arange(
        -1.0 + interval, 1.0, 2 * interval, device=mu.device
    ).repeat(
        (n, c, h, w, 1)
    )  # n c w h maxv
    mu = mu.unsqueeze(-1).repeat((1, 1, 1, 1, size - 1))  # n c w h maxv
    scale = scale.unsqueeze(-1).repeat((1, 1, 1, 1, size - 1))  # n c w h maxv
    invscale = 1.0 / scale
    endpoints_rescaled = (endpoints - mu) * invscale
    cdfs = torch.zeros(n, c, h, w, size + 1, device=mu.device)
    cdfs[..., 1:-1] = torch.sigmoid(endpoints_rescaled)
    cdfs[..., -1] = 1.0
    probs = cdfs[..., 1:] - cdfs[..., :-1]
    probs = modify_regular_prob(probs)
    torch.use_deterministic_algorithms(False)
    cdfs[..., 1:] = torch.cumsum(probs, dim=-1)
    torch.use_deterministic_algorithms(True)
    cdfs[..., -1] = 1.0
    cdfs_q = torch.round(cdfs * float(1 << rans_freq_precision)).to(torch.int16)
    return cdfs_q


def discretized_logistic_prob(
    mu: Tensor, scale: Tensor, x: Tensor, channel_ranges: ColorBitdepths
) -> Tensor:
    """
    function to calculate the log-probability of x under a discretized Logistic(mu, scale) distribution
    heavily based on discretized_mix_logistic_loss() in https://github.com/openai/pixel-cnn

    For each channel we assume that x is in channel_ranges.ranges_int[i]/channel_ranges.scaling_factors[i]
    In case of YCoCg: Y in [0, 1], Co in [-1, 1], Cg in [-1, 1], hence bin size is 1/255
    In case of RGB: R,G,B in [0, 1], hence bin size is 1/255
    """
    max_vs = torch.Tensor(channel_ranges.scaling_factors).to(x.device)[
        None, :, None, None
    ]  # 1 C 1 1
    # FIXME: For now this part is inflexible as it assumes that all channels have the same bin size of 1/255
    bin_sizes = 1.0 / max_vs
    invscale = 1.0 / scale
    x_centered = x - mu

    plus_in = invscale * (x_centered + bin_sizes / 2.0)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = invscale * (x_centered - bin_sizes / 2.0)
    cdf_min = torch.sigmoid(min_in)

    diff = cdf_plus - cdf_min
    threshold = 1 - 1 / max_vs / 2
    # since logistic distribution is leaky, we add the probability mass of the tails to the edge bins
    cond1 = torch.where(x < -threshold, cdf_plus, diff)
    prob = torch.where(x > threshold, torch.ones_like(cdf_min) - cdf_min, cond1)
    return prob


def discretized_logistic_logp(
    mu: Tensor, scale: Tensor, x: Tensor, channel_ranges: ColorBitdepths
) -> Tensor:
    """
    function to calculate the log-probability of x under a discretized Logistic(mu, scale) distribution
    heavily based on discretized_mix_logistic_loss() in https://github.com/openai/pixel-cnn
    x in [0, 1]
    """
    prob = discretized_logistic_prob(mu, scale, x, channel_ranges)
    # no value should cost more than 16 bits
    prob = torch.clamp_min(prob, 2 ** (-16))
    logp = torch.log2(prob)
    return logp


def get_mu_and_scale_colorreg(params: Tensor) -> Tuple[Tensor, Tensor]:
    mu = params[:, ::2, ...]
    log_scale = params[:, 1::2, ...]
    scale = get_scale(log_scale)
    return mu, scale

def weak_colorar_rate(
    mu: Tensor, scale: Tensor, x: Tensor, channel_ranges: ColorBitdepths
) -> Tensor:
    """
    params N 9 H W, x normalized to [0,1]
    """
    logp = discretized_logistic_logp(mu, scale, x, channel_ranges)
    return -logp


def laplace_cdf(x: Tensor, loc: Tensor, scale: Tensor) -> Tensor:
    """Compute the laplace cumulative evaluated in x. All parameters
    must have the same dimension.

    Args:
        x (Tensor): Where the cumulative if evaluated.
        loc (Tensor): Expectation.
        scale (Tensor): Scale

    Returns:
        Tensor: CDF(x, mu, scale)
    """
    return 0.5 - 0.5 * (x - loc).sign() * torch.expm1(-(x - loc).abs() / scale)


def get_latent_rate(
    x: Tensor, mu: Tensor, scale: Tensor, bitdepth: int, freq_precision: int
) -> Tensor:
    """Compute the laplace log-probability evaluated in x. All parameters
    must have the same dimension.

    Args:
        x (Tensor): Where the log-probability if evaluated.
        loc (Tensor): Expectation.
        scale (Tensor): Scale

    Returns:
        Tensor: log(P(x | mu, scale))
    """
    n = 1 << bitdepth
    prob = laplace_cdf(x + 0.5, mu, scale) - laplace_cdf(x - 0.5, mu, scale)

    a = float(2 ** (-freq_precision))
    prob = (1 - n * a) * prob + a

    logp = torch.log2(prob)
    return -logp


def laplace_freqs(
    scale: float, num_symbols: int, freq_precision: int
) -> np.ndarray:
    """
    function to calculate the frequencies of the symbols
    """
    centers = torch.arange(1, num_symbols) - num_symbols / 2
    proba = laplace_cdf(centers + 0.5, 0, scale) - laplace_cdf(
        centers - 0.5, 0, scale
    )
    prob = modify_prob(proba, freq_precision)
    freqs = (prob * (1 << freq_precision)).round().int().unsqueeze(0).numpy()
    return freqs


@torch.no_grad()
def fsar_freqs(
    arm: torch.nn.Module, max_val: int, freq_precision: int, device: str
) -> np.ndarray:
    """
    function to calculate the pmfs of the symbols in order 2
    symbols, 1 dim tensor of symbols
    """
    scale = 1 << freq_precision
    num_symbols = 2 * max_val + 1
    symbols = torch.arange(-max_val, max_val + 1, 1.0).to(device)
    symbols_2d = torch.cartesian_prod(symbols, symbols).float()
    params = arm(symbols_2d)
    mu, scale = get_mu_scale(params)
    mu = mu.repeat(1, num_symbols)
    scale = scale.repeat(1, num_symbols)
    symbols = symbols.repeat(num_symbols**2, 1)
    prob = laplace_cdf(symbols + 0.5, mu, scale) - laplace_cdf(
        symbols - 0.5, mu, scale
    )
    prob = modify_prob(prob, freq_precision)
    freqs = (prob * (1 << freq_precision)).round().int().cpu().numpy()
    return freqs
