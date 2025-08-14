# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.

"""
Lossless loss functions for Cool-Chic

For lossless compression, we only optimize for rate (no distortion term)
The loss is purely the entropy of the pixel distributions.
"""

from typing import Dict, Any
import torch
from torch import Tensor

from lossless.component.synthesis_lossless import logistic_entropy_loss


def lossless_loss_function(
    target_pixels: Tensor,        # [B, C, H, W] ground truth pixels (0-255)
    pixel_mean: Tensor,           # [B, C, H, W] predicted means
    pixel_log_scale: Tensor,      # [B, C, H, W] predicted log-scales  
    rate_latent_bit: Tensor,      # [N] rate of latents in bits
    total_rate_nn_bit: float = 0.0,  # Rate of neural network parameters
    compute_logs: bool = True,
) -> 'LosslessLossFunctionOutput':
    """
    Compute lossless loss function.
    
    For lossless compression, the loss is:
    L = Entropy_loss(pixels | mean, log_scale) + λ_latent * Rate_latent + λ_nn * Rate_nn
    
    Where:
    - Entropy_loss ensures perfect reconstruction capability
    - Rate_latent is the cost of transmitting latent variables
    - Rate_nn is the cost of transmitting neural network parameters
    
    Args:
        target_pixels: Ground truth pixels [B, C, H, W] in range [0, 255]
        pixel_mean: Predicted pixel means [B, C, H, W] 
        pixel_log_scale: Predicted pixel log-scales [B, C, H, W]
        rate_latent_bit: Rate of latent variables in bits [N]
        total_rate_nn_bit: Rate of neural network parameters in bits
        compute_logs: Whether to compute detailed logging information
        
    Returns:
        LosslessLossFunctionOutput containing loss and metrics
    """
    
    # Main entropy loss - this ensures we can reconstruct perfectly
    entropy_loss = logistic_entropy_loss(target_pixels, pixel_mean, pixel_log_scale)
    
    # Rate of latent variables (average over all latents)
    if rate_latent_bit.numel() > 0:
        rate_latent_loss = rate_latent_bit.mean()
    else:
        rate_latent_loss = torch.tensor(0.0, device=target_pixels.device)
    
    # Total loss is entropy + latent rate + network rate
    # For lossless, we weight these differently than lossy compression
    lambda_entropy = 1.0      # Weight for entropy loss
    lambda_latent = 0.01      # Weight for latent rate (smaller than lossy)
    lambda_nn = 0.001         # Weight for network rate
    
    total_loss = (
        lambda_entropy * entropy_loss + 
        lambda_latent * rate_latent_loss + 
        lambda_nn * total_rate_nn_bit
    )
    
    # Compute metrics for logging
    logs = {}
    if compute_logs:
        B, C, H, W = target_pixels.shape
        n_pixels = B * C * H * W
        
        # Convert target pixels to [0, 1] range for calculations
        target_norm = target_pixels / 255.0
        
        # For lossless, we decode deterministically using the mean
        decoded_norm = torch.clamp(pixel_mean, 0.0, 1.0)
        
        # Metrics
        mse = torch.mean((decoded_norm - target_norm) ** 2)
        psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
        
        # Rate calculations  
        total_rate_latent_bpp = rate_latent_bit.sum() / n_pixels if rate_latent_bit.numel() > 0 else 0.0
        total_rate_nn_bpp = total_rate_nn_bit / n_pixels
        total_rate_bpp = total_rate_latent_bpp + total_rate_nn_bpp
        
        logs = {
            "loss": total_loss,
            "entropy_loss": entropy_loss,
            "rate_latent_loss": rate_latent_loss,
            "mse": mse,
            "psnr_db": psnr,
            "total_rate_bpp": total_rate_bpp,
            "total_rate_latent_bpp": total_rate_latent_bpp,
            "total_rate_nn_bpp": total_rate_nn_bpp,
            "lambda_entropy": lambda_entropy,
            "lambda_latent": lambda_latent,
            "lambda_nn": lambda_nn,
        }
    
    return LosslessLossFunctionOutput(
        loss=total_loss,
        entropy_loss=entropy_loss,
        rate_latent_loss=rate_latent_loss,
        logs=logs
    )


class LosslessLossFunctionOutput:
    """Output of lossless loss function."""
    
    def __init__(
        self,
        loss: Tensor,
        entropy_loss: Tensor,
        rate_latent_loss: Tensor,
        logs: Dict[str, Any]
    ):
        self.loss = loss
        self.entropy_loss = entropy_loss
        self.rate_latent_loss = rate_latent_loss
        
        # Add individual components to the object for easy access
        for key, value in logs.items():
            setattr(self, key, value)
        
        # Ensure required attributes exist with default values
        if not hasattr(self, 'psnr_db'):
            self.psnr_db = torch.tensor(0.0)
        if not hasattr(self, 'total_rate_bpp'):
            self.total_rate_bpp = 0.0
        if not hasattr(self, 'total_rate_latent_bpp'):
            self.total_rate_latent_bpp = 0.0
        if not hasattr(self, 'total_rate_nn_bpp'):
            self.total_rate_nn_bpp = 0.0
    
    def pretty_string(self, show_col_name: bool = False, mode: str = "short") -> str:
        """Create a pretty string representation of the results."""
        
        if show_col_name:
            if mode == "short":
                header = f"{'Loss':>8} {'Entropy':>8} {'Rate_L':>8} {'PSNR':>8} {'BPP':>8}"
            else:
                header = f"{'Loss':>8} {'Entropy':>8} {'Rate_L':>8} {'Rate_NN':>8} {'PSNR':>8} {'BPP_L':>8} {'BPP_NN':>8} {'BPP_Tot':>8}"
            return header
        
        if mode == "short":
            return (f"{self.loss.item():>8.4f} "
                   f"{self.entropy_loss.item():>8.4f} "
                   f"{self.rate_latent_loss.item():>8.4f} "
                   f"{self.psnr_db.item() if isinstance(self.psnr_db, Tensor) else self.psnr_db:>8.2f} "
                   f"{self.total_rate_bpp:>8.4f}")
        else:
            return (f"{self.loss.item():>8.4f} "
                   f"{self.entropy_loss.item():>8.4f} "
                   f"{self.rate_latent_loss.item():>8.4f} "
                   f"{self.total_rate_nn_bpp:>8.4f} "
                   f"{self.psnr_db.item() if isinstance(self.psnr_db, Tensor) else self.psnr_db:>8.2f} "
                   f"{self.total_rate_latent_bpp:>8.4f} "
                   f"{self.total_rate_nn_bpp:>8.4f} "
                   f"{self.total_rate_bpp:>8.4f}")


# Example usage and testing
if __name__ == "__main__":
    # Test the lossless loss function
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create test data
    B, C, H, W = 1, 3, 32, 32
    target_pixels = torch.randint(0, 256, (B, C, H, W)).float().to(device)
    pixel_mean = torch.randn(B, C, H, W).to(device) * 0.1 + 0.5  # Around 0.5
    pixel_log_scale = torch.randn(B, C, H, W).to(device) * 0.1 - 2.0  # Around -2.0
    rate_latent_bit = torch.randn(100).abs().to(device) * 5.0  # Some rate values
    
    # Compute loss
    result = lossless_loss_function(
        target_pixels=target_pixels,
        pixel_mean=pixel_mean,
        pixel_log_scale=pixel_log_scale,
        rate_latent_bit=rate_latent_bit,
        total_rate_nn_bit=1000.0,  # Some network rate
        compute_logs=True
    )
    
    print("Lossless Loss Function Test:")
    print("=" * 50)
    print(result.pretty_string(show_col_name=True, mode="long"))
    print(result.pretty_string(show_col_name=False, mode="long"))
    
    print(f"\nDetailed metrics:")
    print(f"Total loss: {result.loss.item():.4f}")
    print(f"Entropy loss: {result.entropy_loss.item():.4f}")
    print(f"Rate latent loss: {result.rate_latent_loss.item():.4f}")
    print(f"PSNR: {result.psnr_db.item() if isinstance(result.psnr_db, Tensor) else result.psnr_db:.2f} dB")
    print(f"Total rate: {result.total_rate_bpp:.4f} bpp")
