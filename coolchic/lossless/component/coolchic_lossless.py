# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.

"""
Lossless CoolChicEncoder - Modified version for lossless compression

Key changes from lossy version:
1. Uses LosslessSynthesis that outputs distribution parameters
2. Forward pass returns both distribution parameters and decoded pixels
3. Designed for entropy coding workflow
"""

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, OrderedDict, Tuple
from typing_extensions import TypedDict
import torch.nn.functional as F

import torch
from torch import Tensor, nn

# Import from existing Cool-Chic modules
from enc.component.core.arm import Arm, _get_neighbor, _get_non_zero_pixel_ctx_index
from enc.component.core.quantizer import (
    POSSIBLE_QUANTIZATION_NOISE_TYPE,
    POSSIBLE_QUANTIZER_TYPE,
    quantize,
)
from enc.component.core.upsampling import Upsampling
from enc.utils.device import POSSIBLE_DEVICE
from enc.component.types import DescriptorCoolChic, DescriptorNN

# Import our new lossless synthesis
from lossless.component.synthesis_lossless import LosslessSynthesis, logistic_entropy_loss, entropy_decode_logistic


def _laplace_cdf(input: Tensor, mu: Tensor, scale: Tensor) -> Tensor:
    """Cumulative density function of the laplace distribution."""
    return 0.5 - 0.5 * (input - mu).sign() * torch.expm1(-(input - mu).abs() / scale)


@dataclass
class LosslessCoolChicEncoderParameter:
    """Parameters for lossless CoolChicEncoder - similar to original but adapted for lossless."""
    
    # Image parameters
    img_size: Tuple[int, int] = (512, 768)
    
    # Latent parameters
    latent_n_grids: int = 7
    n_ft_per_res: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1, 1])
    
    # Encoder gains
    encoder_gains: List[float] = field(default_factory=lambda: [16.0])
    
    # ARM parameters  
    dim_arm: int = 16
    n_hidden_layers_arm: int = 2
    
    # Upsampling parameters
    ups_k_size: int = 8
    ups_preconcat_k_size: int = 7
    
    # Synthesis parameters (same format as original)
    layers_synthesis: List[str] = field(default_factory=lambda: [
        "48-1-linear-relu",
        "X-1-linear-none"  # Will output 6 channels for RGB (3 mean + 3 log_scale)
    ])
    
    def set_image_size(self, img_size: Tuple[int, int]) -> None:
        """Update image size."""
        self.img_size = img_size


class LosslessCoolChicEncoderOutput(TypedDict):
    """Output of lossless CoolChicEncoder forward pass."""
    
    # Distribution parameters
    pixel_mean: Tensor      # [B, C, H, W] - predicted means
    pixel_log_scale: Tensor # [B, C, H, W] - predicted log-scales
    
    # Decoded pixels (from distribution parameters)
    decoded_pixels: Tensor  # [B, C, H, W] - sampled/decoded pixels
    
    # Rate information
    rate: Tensor           # [N] - bits per latent
    
    # Additional data for analysis
    additional_data: Dict[str, Any]


class LosslessCoolChicEncoder(nn.Module):
    """Lossless version of CoolChicEncoder."""

    def __init__(self, param: LosslessCoolChicEncoderParameter):
        super().__init__()
        
        self.param = param
        
        # ==================== Latent grids ==================== #
        assert len(self.param.n_ft_per_res) == self.param.latent_n_grids, (
            f"len(n_ft_per_res) = {len(self.param.n_ft_per_res)} should be equal to "
            f"latent_n_grids = {self.param.latent_n_grids}"
        )

        # Encoder gain
        self.encoder_gains = nn.Parameter(
            torch.FloatTensor(self.param.encoder_gains), requires_grad=False
        )

        # Create latent grids
        self.size_per_latent = []
        self.latent_grids = nn.ParameterList()
        for i in range(self.param.latent_n_grids):
            h_grid, w_grid = [int(math.ceil(x / (2**i))) for x in self.param.img_size]
            c_grid = self.param.n_ft_per_res[i]
            cur_size = (1, c_grid, h_grid, w_grid)

            self.size_per_latent.append(cur_size)
            self.latent_grids.append(
                nn.Parameter(torch.empty(cur_size), requires_grad=True)
            )

        self.initialize_latent_grids()
        # ==================== Latent grids ==================== #

        # ==================== Synthesis ==================== #
        # Use lossless synthesis that outputs distribution parameters
        self.synthesis = LosslessSynthesis(
            input_ft=sum([latent_size[1] for latent_size in self.size_per_latent]),
            layers_dim=self.param.layers_synthesis,
            n_output_channels=3  # RGB
        )
        # ==================== Synthesis ==================== #

        # ==================== Upsampling ==================== #
        self.upsampling = Upsampling(
            ups_k_size=self.param.ups_k_size,
            ups_preconcat_k_size=self.param.ups_preconcat_k_size,
            n_ups_kernel=self.param.latent_n_grids - 1,
            n_ups_preconcat_kernel=self.param.latent_n_grids - 1,
        )
        # ==================== Upsampling ==================== #

        # ==================== ARM ==================== #
        max_mask_size = 9
        max_context_pixel = int((max_mask_size**2 - 1) / 2)
        assert self.param.dim_arm <= max_context_pixel, (
            f"ARM dimension ({self.param.dim_arm}) too large"
        )

        self.mask_size = max_mask_size
        self.register_buffer(
            "non_zero_pixel_ctx_index",
            _get_non_zero_pixel_ctx_index(self.param.dim_arm),
            persistent=False,
        )

        self.arm = Arm(self.param.dim_arm, self.param.n_hidden_layers_arm)
        # ==================== ARM ==================== #

        # Monitoring and quantization tracking
        self.modules_to_send = ["arm", "synthesis", "upsampling"] 
        self.nn_q_step: Dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }
        self.nn_expgol_cnt: Dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }
        self.full_precision_param = None

    def initialize_latent_grids(self) -> None:
        """Initialize latent grids with random values."""
        for latent_grid in self.latent_grids:
            nn.init.uniform_(latent_grid, -0.5, 0.5)

    def forward(
        self,
        target_pixels: Optional[Tensor] = None,  # Ground truth for training
        quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "kumaraswamy",
        quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
        soft_round_temperature: Optional[Tensor] = torch.tensor(0.3),
        noise_parameter: Optional[Tensor] = torch.tensor(1.0),
        AC_MAX_VAL: int = -1,
        flag_additional_outputs: bool = False,
        deterministic_decode: bool = True,
    ) -> LosslessCoolChicEncoderOutput:
        """
        Forward pass for lossless compression.
        
        Args:
            target_pixels: Ground truth pixels [B, C, H, W] for training (0-255 range)
            deterministic_decode: If True, use mean for decoding; if False, sample
            Other args: Same as original CoolChicEncoder
            
        Returns:
            LosslessCoolChicEncoderOutput with distribution parameters and decoded pixels
        """
        
        # ------ Step 1: Quantize latents (same as original) ------
        encoder_side_flat_latent = torch.cat(
            [latent_i.view(-1) for latent_i in self.latent_grids]
        )

        flat_decoder_side_latent = quantize(
            encoder_side_flat_latent * self.encoder_gains,
            quantizer_noise_type if self.training else "none",
            quantizer_type if self.training else "hardround",
            soft_round_temperature,
            noise_parameter,
        )

        if AC_MAX_VAL != -1:
            flat_decoder_side_latent = torch.clamp(
                flat_decoder_side_latent, -AC_MAX_VAL, AC_MAX_VAL + 1
            )

        # Convert back to list of latent grids
        decoder_side_latent = []
        cnt = 0
        for latent_size in self.size_per_latent:
            b, c, h, w = latent_size
            latent_numel = b * c * h * w
            decoder_side_latent.append(
                flat_decoder_side_latent[cnt : cnt + latent_numel].view(latent_size)
            )
            cnt += latent_numel

        # ------ Step 2: ARM to estimate rate (same as original) ------
        flat_context = torch.cat(
            [
                _get_neighbor(
                    spatial_latent_i, self.mask_size, self.non_zero_pixel_ctx_index
                )
                for spatial_latent_i in decoder_side_latent
            ],
            dim=0,
        )

        flat_latent = torch.cat(
            [spatial_latent_i.view(-1) for spatial_latent_i in decoder_side_latent],
            dim=0,
        )

        flat_mu, flat_scale, flat_log_scale = self.arm(flat_context)

        # Compute rate using Laplace distribution
        proba = torch.clamp_min(
            _laplace_cdf(flat_latent + 0.5, flat_mu, flat_scale)
            - _laplace_cdf(flat_latent - 0.5, flat_mu, flat_scale),
            min=2**-16,
        )
        flat_rate = -torch.log2(proba)

        # ------ Step 3: Upsampling and lossless synthesis ------
        ups_out = self.upsampling(decoder_side_latent)
        
        # Get distribution parameters instead of direct pixels
        pixel_mean, pixel_log_scale = self.synthesis(ups_out)
        
        # Upsample to target resolution if needed
        pixel_mean = F.interpolate(pixel_mean, size=self.param.img_size, mode="nearest")
        pixel_log_scale = F.interpolate(pixel_log_scale, size=self.param.img_size, mode="nearest")

        # ------ Step 4: Decode pixels from distribution ------
        if deterministic_decode:
            # Use mean for deterministic decoding
            decoded_pixels = pixel_mean
        else:
            # Sample from logistic distribution
            decoded_pixels = entropy_decode_logistic(pixel_mean, pixel_log_scale, num_samples=1)

        # Clamp decoded pixels to valid range
        decoded_pixels = torch.clamp(decoded_pixels, 0.0, 1.0)

        # ------ Step 5: Prepare additional data ------
        additional_data = {}
        if flag_additional_outputs:
            additional_data["upsampled_latent"] = ups_out
            additional_data["flat_rate"] = flat_rate
            additional_data["flat_mu"] = flat_mu
            additional_data["flat_scale"] = flat_scale
            
            # If we have target pixels, compute entropy loss
            if target_pixels is not None:
                entropy_loss = logistic_entropy_loss(target_pixels, pixel_mean, pixel_log_scale)
                additional_data["entropy_loss"] = entropy_loss

        return LosslessCoolChicEncoderOutput(
            pixel_mean=pixel_mean,
            pixel_log_scale=pixel_log_scale,
            decoded_pixels=decoded_pixels,
            rate=flat_rate,
            additional_data=additional_data,
        )

    def get_param(self) -> OrderedDict[str, Tensor]:
        """Get model parameters."""
        return self.state_dict()

    def set_param(self, param: OrderedDict[str, Tensor]):
        """Set model parameters."""
        self.load_state_dict(param)

    def reinitialize_parameters(self) -> None:
        """Reinitialize all parameters."""
        self.arm.reinitialize_parameters()
        self.upsampling.reinitialize_parameters()
        self.synthesis.reinitialize_parameters()
        self.initialize_latent_grids()

    def to_device(self, device: POSSIBLE_DEVICE) -> None:
        """Move model to device."""
        self.to(device)


# Example usage
if __name__ == "__main__":
    # Test the lossless encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create encoder with small image for testing
    param = LosslessCoolChicEncoderParameter(
        img_size=(64, 64),
        latent_n_grids=3,  # Smaller for testing
        n_ft_per_res=[1, 1, 1],
        layers_synthesis=[
            "16-1-linear-relu",
            "X-1-linear-none"  # Will output 6 channels for RGB
        ]
    )
    
    encoder = LosslessCoolChicEncoder(param).to(device)
    
    # Create dummy target pixels (RGB image with values 0-255)
    target_pixels = torch.randint(0, 256, (1, 3, 64, 64)).float().to(device)
    
    # Forward pass
    output = encoder(
        target_pixels=target_pixels,
        flag_additional_outputs=True,
        deterministic_decode=True
    )
    
    print(f"Target pixels shape: {target_pixels.shape}")
    print(f"Pixel mean shape: {output['pixel_mean'].shape}")
    print(f"Pixel log-scale shape: {output['pixel_log_scale'].shape}")
    print(f"Decoded pixels shape: {output['decoded_pixels'].shape}")
    print(f"Rate shape: {output['rate'].shape}")
    
    if "entropy_loss" in output["additional_data"]:
        print(f"Entropy loss: {output['additional_data']['entropy_loss'].item():.4f}")
    
    # Convert decoded pixels to 0-255 range for comparison
    decoded_255 = (output['decoded_pixels'] * 255).clamp(0, 255)
    target_255 = target_pixels
    
    print(f"Mean absolute error: {(decoded_255 - target_255).abs().mean():.2f}")
