import torch
import constriction
import struct


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


def calculate_laplace_probability_distribution(mu, b):
    """Calculate Laplace probability distribution for arithmetic coding.
    Works for any shape of mu and b, adds one dimension at the end for the probability axis.
    """
    # Create the base tensor of quantized values
    new_tensor = torch.linspace(0, 255, steps=256, device=mu.device)  # [256]
    new_shape = (*mu.shape, 256)  # add one dimension at the end
    new_tensor = new_tensor.view(*([1] * mu.ndim), 256).expand(*new_shape)

    # Compute boundaries for each bin
    x_minus = new_tensor - 0.5
    x_plus = new_tensor + 0.5

    # Expand mu and b with one trailing dimension
    mu_expanded = mu.unsqueeze(-1)
    b_expanded = b.unsqueeze(-1)

    # Laplace CDF difference between bin edges (use the laplace cdf function)
    cdf_minus = _laplace_cdf(x_minus, mu_expanded, b_expanded)
    cdf_plus = _laplace_cdf(x_plus, mu_expanded, b_expanded)

    prob_t = cdf_plus - cdf_minus
    prob_t = torch.clamp_min(prob_t, 2 ** (-16))
    prob_t = prob_t / prob_t.sum(dim=-1, keepdim=True)

    return prob_t

def calculate_logistic_probability_distribution(mu, s):
    """Calculate Logistic probability distribution for arithmetic coding.
    Works for any shape of mu and s, adds one dimension at the end for the probability axis.
    """
    # Create the base tensor of quantized values
    new_tensor = torch.linspace(0, 255/256, steps=256, device=mu.device)  # [256]
    new_shape = (*mu.shape, 256)  # add one dimension at the end
    new_tensor = new_tensor.view(*([1] * mu.ndim), 256).expand(*new_shape)

    # Compute boundaries for each bin
    x_minus = new_tensor - 0.5 / 256
    x_plus = new_tensor + 0.5 / 256

    # Expand mu and s with one trailing dimension
    mu_expanded = mu.unsqueeze(-1)
    s_expanded = s.unsqueeze(-1)

    # Logistic CDF difference between bin edges (use the logistic cdf function)
    cdf_minus = _logistic_cdf(x_minus, mu_expanded, s_expanded)
    cdf_plus = _logistic_cdf(x_plus, mu_expanded, s_expanded)

    prob_t = cdf_plus - cdf_minus
    prob_t = torch.clamp_min(prob_t, 2 ** (-16))
    prob_t = prob_t / prob_t.sum(dim=-1, keepdim=True)

    return prob_t


def encode(
    x: torch.Tensor,
    mu: torch.Tensor,
    scale: torch.Tensor,
    output_path="./test-workdir/encoder_size_test/coolchic_encoded.binary",
):
    # x_maxv = (1 << 8) - 1
    # x = torch.round(x_maxv * x).to(torch.int8).float() / x_maxv
    # x_reshape = torch.round(x * ((1 << 8) - 1)).to(torch.int8).cpu()
    # x = x * 256.0
    # x = torch.round(x).clamp(0, 255)
    # x_reshape = x.to(torch.int16).cpu()
    
    B, C, H, W = x.shape
    enc = constriction.stream.stack.AnsCoder()
    bits_theoretical = 0

    with torch.no_grad():
        scale_theoretical_bits = 0
        mu = mu.flatten(2, 3).permute(0, 2, 1)  # B, H*W, C
        scale = scale.flatten(2, 3).permute(0, 2, 1)  # B, H*W, C

        prob = calculate_laplace_probability_distribution(mu, scale)
        # print(prob.shape)
        prob_logistic = calculate_logistic_probability_distribution(mu, scale)
        # print(prob_logistic.shape)

        scale_theoretical_bits = 0
        for wh in range(x.shape[2] * x.shape[3]):
            for c in range(x.shape[1]):
                sym: int = (
                    (
                        x_reshape.flatten(2, 3).permute(0, 2, 1)[
                            0, -wh - 1, -c - 1
                        ] * 256
                    )
                    .int()
                    .item()
                )
                prob_t = prob_logistic[0, -wh - 1, -c - 1]
                scale_theoretical_bits += -torch.log2(prob_t[sym]).item()
                model = constriction.stream.model.Categorical(
                    prob_t.detach().cpu().numpy(), perfect=False
                )
                enc.encode_reverse(sym, model)
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
