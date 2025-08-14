# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.

"""
Lossless Synthesis Module for Cool-Chic

This replaces the standard synthesis module to output distribution parameters
(mean and log-scale) for logistic distributions instead of direct pixel values.
"""

import math
from typing import List, OrderedDict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LosslessSynthesisConv2d(nn.Module):
    """Convolution layer for lossless synthesis - similar to standard SynthesisConv2d 
    but designed for distribution parameter prediction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        residual: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.residual = residual

        # Standard convolution
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the layer weights with Xavier/Glorot initialization."""
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with optional residual connection."""
        out = self.conv(x)
        
        if self.residual:
            # For residual connections, input and output channels must match
            assert self.in_channels == self.out_channels, (
                f"Residual connection requires matching channels: "
                f"in={self.in_channels}, out={self.out_channels}"
            )
            out = out + x
            
        return out


class LosslessSynthesis(nn.Module):
    """
    Lossless synthesis module that outputs distribution parameters instead of pixels.
    
    The key difference from standard synthesis:
    - Last layer outputs 2 * n_channels features (mean + log_scale for each channel)
    - No direct pixel prediction, but parameters for entropy coding
    """

    possible_non_linearity = {
        "none": nn.Identity,
        "relu": nn.ReLU,
    }

    possible_mode = ["linear", "residual"]

    def __init__(self, input_ft: int, layers_dim: List[str], n_output_channels: int = 3):
        """
        Args:
            input_ft: Number of input features from upsampling
            layers_dim: Description of each synthesis layer (same format as original)
            n_output_channels: Number of image channels (3 for RGB, 1 for grayscale)
        """
        super().__init__()

        self.input_ft = input_ft
        self.n_output_channels = n_output_channels
        self.layers = nn.ModuleList()

        # Build the synthesis network
        current_channels = input_ft
        
        for i, layer_desc in enumerate(layers_dim):
            out_ft, k_size, mode, non_linearity = layer_desc.split("-")
            
            # Replace 'X' with the correct number of output features
            if out_ft == "X":
                if i == len(layers_dim) - 1:  # Last layer
                    # Output mean and log_scale for each channel
                    out_ft = 2 * n_output_channels
                else:
                    # For intermediate layers, use a reasonable number
                    out_ft = max(current_channels, n_output_channels)
            
            out_ft = int(out_ft)
            k_size = int(k_size)

            # Validation
            assert mode in self.possible_mode, f"Unknown mode: {mode}"
            assert non_linearity in self.possible_non_linearity, f"Unknown non_linearity: {non_linearity}"

            # Add convolution layer
            self.layers.append(
                LosslessSynthesisConv2d(
                    current_channels, out_ft, k_size, residual=(mode == "residual")
                )
            )
            
            # Add non-linearity (except for last layer)
            if i < len(layers_dim) - 1:  # Not the last layer
                self.layers.append(self.possible_non_linearity[non_linearity]())
            
            current_channels = out_ft

        # Ensure the final output has the right number of channels
        assert current_channels == 2 * n_output_channels, (
            f"Final layer must output {2 * n_output_channels} channels "
            f"(mean + log_scale for {n_output_channels} image channels), "
            f"but got {current_channels}"
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass that outputs distribution parameters.
        
        Args:
            x: Input tensor [B, C_in, H, W] from upsampling
            
        Returns:
            Tuple of (mean, log_scale) tensors, each [B, n_output_channels, H, W]
        """
        # Pass through all layers
        for layer in self.layers:
            x = layer(x)
        
        # Split output into mean and log_scale
        # x has shape [B, 2*n_output_channels, H, W]
        mean = x[:, :self.n_output_channels, :, :]
        log_scale = x[:, self.n_output_channels:, :, :]
        
        # Ensure log_scale is not too negative (for numerical stability)
        log_scale = torch.clamp(log_scale, min=-10.0, max=10.0)
        
        return mean, log_scale

    def reinitialize_parameters(self) -> None:
        """Reinitialize all parameters."""
        for module in self.layers:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()


def logistic_entropy_loss(pixels: Tensor, mean: Tensor, log_scale: Tensor) -> Tensor:
    """
    Compute the entropy loss for logistic distribution.
    
    Args:
        pixels: Ground truth pixels [B, C, H, W] (integer values 0-255)
        mean: Predicted mean [B, C, H, W]
        log_scale: Predicted log-scale [B, C, H, W]
        
    Returns:
        Entropy loss (negative log-likelihood)
    """
    scale = torch.exp(log_scale)
    
    # Convert pixels to float and normalize to [0, 1]
    pixels_norm = pixels.float() / 255.0
    
    # Logistic CDF for continuous distribution
    # P(X <= x) = sigmoid((x - mean) / scale)
    # For discrete case: P(X = x) = P(x-0.5 <= X <= x+0.5)
    upper_cdf = torch.sigmoid((pixels_norm + 0.5/255.0 - mean) / scale)
    lower_cdf = torch.sigmoid((pixels_norm - 0.5/255.0 - mean) / scale)
    
    # Probability mass for this pixel value
    prob = upper_cdf - lower_cdf
    prob = torch.clamp(prob, min=1e-7, max=1.0)  # Avoid log(0)
    
    # Negative log-likelihood
    nll = -torch.log(prob)
    
    return nll.mean()


def entropy_decode_logistic(mean: Tensor, log_scale: Tensor, num_samples: int = 1) -> Tensor:
    """
    Sample from logistic distribution for decoding.
    
    Args:
        mean: Predicted mean [B, C, H, W]
        log_scale: Predicted log-scale [B, C, H, W]
        num_samples: Number of samples (for deterministic decoding, use 1)
        
    Returns:
        Decoded pixels [B, C, H, W] as float in [0, 1]
    """
    scale = torch.exp(log_scale)
    
    if num_samples == 1:
        # Deterministic decoding: use the mean
        decoded = mean
    else:
        # Sample from logistic distribution
        # Logistic distribution: F^(-1)(u) = mean + scale * log(u / (1 - u))
        u = torch.rand_like(mean)
        decoded = mean + scale * torch.log(u / (1 - u))
    
    # Clamp to valid range
    decoded = torch.clamp(decoded, 0.0, 1.0)
    
    return decoded


# Example usage and testing
if __name__ == "__main__":
    # Test the lossless synthesis module
    batch_size = 1
    height, width = 64, 64
    input_features = 32
    
    # Create test input
    x = torch.randn(batch_size, input_features, height, width)
    
    # Test synthesis architecture (similar to your config)
    layers_dim = [
        "48-1-linear-relu",
        "X-1-linear-none",  # This will become 6 channels for RGB (3 mean + 3 log_scale)
    ]
    
    # Create lossless synthesis
    synthesis = LosslessSynthesis(
        input_ft=input_features,
        layers_dim=layers_dim,
        n_output_channels=3  # RGB
    )
    
    # Forward pass
    mean, log_scale = synthesis(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Mean shape: {mean.shape}")
    print(f"Log-scale shape: {log_scale.shape}")
    
    # Test entropy loss with dummy ground truth
    gt_pixels = torch.randint(0, 256, (batch_size, 3, height, width))
    loss = logistic_entropy_loss(gt_pixels, mean, log_scale)
    print(f"Entropy loss: {loss.item():.4f}")
    
    # Test decoding
    decoded = entropy_decode_logistic(mean, log_scale)
    print(f"Decoded shape: {decoded.shape}")
    print(f"Decoded range: [{decoded.min():.3f}, {decoded.max():.3f}]")
