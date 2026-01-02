import torch
from lossless.util.color_transform import ColorBitdepths


def _laplace_cdf(x: torch.Tensor, expectation: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
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
    return 0.5 - 0.5 * (shifted_x).sign() * torch.expm1(-(shifted_x).abs() / scale)


def _logistic_cdf(x: torch.Tensor, mu: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
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
        (color_bitdepths.ranges_int[channel_idx][1] - color_bitdepths.ranges_int[channel_idx][0])
        / color_bitdepths.scaling_factors[channel_idx],
        steps=color_bitdepths.bins[channel_idx],
        device=mu.device,
    )
    new_shape = (
        *mu.shape,
        color_bitdepths.bins[channel_idx],
    )  # add one dimension at the end
    new_tensor = new_tensor.view(*([1] * mu.ndim), color_bitdepths.bins[channel_idx]).expand(
        *new_shape
    )

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
    elif distribution == "logistic":
        cdf_minus = _logistic_cdf(x_minus, mu_expanded, s_expanded)
        cdf_plus = _logistic_cdf(x_plus, mu_expanded, s_expanded)
    elif distribution == "dummy":
        cdf_plus = torch.ones_like(x_plus)
        cdf_minus = torch.zeros_like(x_minus)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    prob_t = cdf_plus - cdf_minus
    prob_t = torch.clamp_min(prob_t, 2 ** (-16))
    prob_t = prob_t / prob_t.sum(dim=-1, keepdim=True)

    assert torch.all(
        torch.isclose(prob_t.sum(dim=-1), torch.ones_like(mu))
    ), "Probabilities do not sum to 1"
    assert torch.all(prob_t.min() > 0), "Some probabilities are zero"
    return prob_t