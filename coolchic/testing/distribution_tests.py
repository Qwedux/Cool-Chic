import os
import sys
os.chdir(os.path.dirname(os.getcwd()))
sys.path.append(os.getcwd())

import unittest
from lossless.util.distribution import weak_colorar_rate
import lossless.util.color_transform as color_transform
import torch

def unget_scale(scale:torch.Tensor) -> torch.Tensor:
    """Helper function to get log_scale from scale

    Args:
        - scale (torch.Tensor): Tensor of shape N x C x H x W
    """
    logscale = torch.log(scale) * - 2
    logscale = torch.clamp(logscale, min=-10.0, max=13.8155)
    return logscale

def inflate_mu_and_scale_linear_color(mu:torch.Tensor, log_scale:torch.Tensor, x:torch.Tensor) -> torch.Tensor:
    """Helper function to get params from mu and scale

    Args:
        - mu (torch.Tensor): Tensor of shape N x 3 x H x W
        - log_scale (torch.Tensor): Tensor of shape N x 3 x H x W
        - x (torch.Tensor): Tensor of shape N x 3 x H x W
    """
    raise NotImplementedError("Function not implemented yet.")


class TestWeakColorarRate(unittest.TestCase):
    def simple_case(self):
        channel_ranges = color_transform.RGBBitdepths()
        

if __name__ == "__main_":
    unittest.main()