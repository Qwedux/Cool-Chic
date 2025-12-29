import torch
import torch.nn as nn
from lossless.util.distribution import get_scale


def get_mu_and_log_scale_linear_color(
    params: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    function to calculate the mean and scale from the parameters
    """
    # in the 1st dimension we have the following structure:
    # [0     , 1        , 2       , 3          , 4    , 5      , 6         , 7    , 8   ]
    # [mu_red, scale_red, mu_green, scale_green, alpha, mu_blue, scale_blue, beta, gamma]
    _mu = torch.cat([params[:, 0:1, ...], params[:, 2:3, ...], params[:, 5:6, ...]], dim=1)
    log_scale = torch.cat([params[:, 1:2, ...], params[:, 3:4, ...], params[:, 6:7, ...]], dim=1)
    pp = torch.cat([params[:, 4:5, ...], params[:, 7:8, ...], params[:, 8:9, ...]], dim=1)
    alpha, beta, gamma = torch.chunk(pp, 3, dim=1)

    mu = torch.zeros_like(_mu)
    mu[:, 0:1, ...] = _mu[:, 0:1, ...]
    mu[:, 1:2, ...] = _mu[:, 1:2, ...] + alpha * x[:, 0:1, ...]
    mu[:, 2:3, ...] = _mu[:, 2:3, ...] + beta * x[:, 0:1, ...] + gamma * x[:, 1:2, ...]
    return mu, log_scale


class ProbabilityOutput(nn.Module):
    """Instantiate a output layer that gets distribution params (mu, scale)
    from model outputs. Input tensor :math:`\\mathbf{x}` with shape :math:`[B, C_{in}, H, W]`, producing
    an output tensors :math:`\\mathbf{mu, scale}` with shape :math:`[B, 6, H, W]`.

    .. math::

        \\mathbf{(mu, scale)} =
        \\begin{cases}
            \\mathrm{colorreg}(\\mathbf{x}) & \\text{if color regression,} \\\\
            \\bigl(\\mathbf{x}_{:,0::2,\\ldots}, \\mathbf{x}_{:,1::2,\\ldots}
            \\bigr) & \\text{otherwise.} \\\\
        \\end{cases}
    """

    def __init__(
        self,
        do_color_regression: bool = False,
    ):
        """
        Args:
            do_color_regression: Whether to use color regression to get mu and scale.
                Default to False.
        """
        super().__init__()

        self.do_color_regression = do_color_regression

    def forward(
        self, network_out: torch.Tensor, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass of this layer.

        Args:
            network_out: Input tensor of shape :math:`[B, C_{in}, H, W]`.
            image: Input image tensor of shape :math:`[B, C_{in}, H, W]`.

        Returns:
            (mu, scale) of shape :math:`[B, 3, H, W]`.
        """
        if self.do_color_regression:
            # this is the ugly case of doing color regression
            mu, log_scale = get_mu_and_log_scale_linear_color(network_out, image)
        else:
            mu = network_out[:, 0::2, ...]
            log_scale = network_out[:, 1::2, ...]

        return mu, get_scale(log_scale)
