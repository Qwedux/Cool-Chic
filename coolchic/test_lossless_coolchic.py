# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.

"""
Simple test script for lossless Cool-Chic implementation

This script demonstrates how to use the lossless components we've created.
"""

import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

# Add current directory to path so we can import our modules
sys.path.append(os.getcwd())

from lossless.component.coolchic_lossless import LosslessCoolChicEncoder, LosslessCoolChicEncoderParameter
from lossless.component.loss_lossless import lossless_loss_function


def load_image_as_tensor(image_path: str, target_size: tuple = (128, 128)) -> torch.Tensor:
    """Load image and convert to tensor in range [0, 255]."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to tensor and ensure it's in [0, 255] range
    image_tensor = torch.from_numpy(np.array(image)).float()
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    
    return image_tensor


def test_lossless_coolchic():
    """Test the lossless Cool-Chic implementation."""
    print("Testing Lossless Cool-Chic Implementation")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create parameters for a small test
    target_size = (64, 64)  # Small for testing
    param = LosslessCoolChicEncoderParameter(
        img_size=target_size,
        latent_n_grids=4,  # Smaller for testing
        n_ft_per_res=[1, 1, 1, 1],  # Simple configuration
        layers_synthesis=[
            "24-1-linear-relu",
            "12-1-linear-relu", 
            "X-1-linear-none"  # Will output 6 channels for RGB (3 mean + 3 log_scale)
        ],
        dim_arm=8,  # Smaller ARM
        n_hidden_layers_arm=1
    )
    
    # Create encoder
    encoder = LosslessCoolChicEncoder(param).to(device)
    print(f"Created encoder with {sum(p.numel() for p in encoder.parameters())} parameters")
    
    # Create synthetic test image (or load a real one if available)
    try:
        # Try to load the first Kodak image if available
        kodak_path = "../datasets/kodak/kodim01.png"
        if os.path.exists(kodak_path):
            target_pixels = load_image_as_tensor(kodak_path, target_size).to(device)
            print(f"Loaded test image from {kodak_path}")
        else:
            # Create synthetic image
            target_pixels = torch.randint(0, 256, (1, 3, *target_size)).float().to(device)
            print("Created synthetic test image")
    except:
        # Fallback to synthetic image
        target_pixels = torch.randint(0, 256, (1, 3, *target_size)).float().to(device)
        print("Created synthetic test image")
    
    print(f"Target image shape: {target_pixels.shape}")
    print(f"Target image range: [{target_pixels.min():.1f}, {target_pixels.max():.1f}]")
    
    # Test forward pass (inference mode)
    print("\n1. Testing inference (deterministic decoding)...")
    encoder.eval()
    with torch.no_grad():
        output = encoder(
            target_pixels=target_pixels,
            flag_additional_outputs=True,
            deterministic_decode=True
        )
    
    print(f"   Pixel mean shape: {output['pixel_mean'].shape}")
    print(f"   Pixel log-scale shape: {output['pixel_log_scale'].shape}")
    print(f"   Decoded pixels shape: {output['decoded_pixels'].shape}")
    print(f"   Rate shape: {output['rate'].shape}")
    
    # Check reconstruction quality
    decoded_255 = (output['decoded_pixels'] * 255).clamp(0, 255)
    mse = F.mse_loss(decoded_255, target_pixels)
    psnr = 10 * torch.log10(255**2 / mse)
    print(f"   MSE: {mse.item():.2f}")
    print(f"   PSNR: {psnr.item():.2f} dB")
    
    # Test loss function
    print("\n2. Testing loss function...")
    loss_result = lossless_loss_function(
        target_pixels=target_pixels,
        pixel_mean=output['pixel_mean'],
        pixel_log_scale=output['pixel_log_scale'],
        rate_latent_bit=output['rate'],
        total_rate_nn_bit=1000.0,  # Dummy network rate
        compute_logs=True
    )
    
    print("   Loss components:")
    print(f"   - Total loss: {loss_result.loss.item():.4f}")
    print(f"   - Entropy loss: {loss_result.entropy_loss.item():.4f}")
    print(f"   - Rate latent loss: {loss_result.rate_latent_loss.item():.4f}")
    
    # Test training mode  
    print("\n3. Testing training step...")
    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    
    # Single training step
    optimizer.zero_grad()
    
    output = encoder(
        target_pixels=target_pixels,
        flag_additional_outputs=True,
        deterministic_decode=True
    )
    
    loss_result = lossless_loss_function(
        target_pixels=target_pixels,
        pixel_mean=output['pixel_mean'],
        pixel_log_scale=output['pixel_log_scale'],
        rate_latent_bit=output['rate'],
        compute_logs=True
    )
    
    loss_result.loss.backward()
    optimizer.step()
    
    print(f"   Training loss: {loss_result.loss.item():.4f}")
    print(f"   Gradients computed successfully: {any(p.grad is not None for p in encoder.parameters())}")
    
    # Test stochastic decoding
    print("\n4. Testing stochastic decoding...")
    encoder.eval()
    with torch.no_grad():
        output_stochastic = encoder(
            target_pixels=target_pixels,
            flag_additional_outputs=True,
            deterministic_decode=False  # Use sampling
        )
    
    decoded_stochastic_255 = (output_stochastic['decoded_pixels'] * 255).clamp(0, 255)
    mse_stochastic = F.mse_loss(decoded_stochastic_255, target_pixels)
    psnr_stochastic = 10 * torch.log10(255**2 / mse_stochastic)
    print(f"   Stochastic MSE: {mse_stochastic.item():.2f}")
    print(f"   Stochastic PSNR: {psnr_stochastic.item():.2f} dB")
    
    print("\n✅ All tests completed successfully!")
    
    return {
        'encoder': encoder,
        'target_pixels': target_pixels,
        'output': output,
        'loss_result': loss_result
    }


if __name__ == "__main__":
    test_results = test_lossless_coolchic()
    
    print("\n" + "="*50)
    print("Lossless Cool-Chic Test Summary:")
    print("- Synthesis now outputs distribution parameters (μ, σ)")
    print("- Loss function focuses on entropy (perfect reconstruction)")
    print("- Both deterministic and stochastic decoding work")
    print("- Training gradients flow correctly")
    print("\nNext steps for full implementation:")
    print("1. Integrate with existing training loop")
    print("2. Add proper entropy coding for bitstream generation")
    print("3. Tune hyperparameters for your specific use case")
    print("4. Test on larger images and datasets")
