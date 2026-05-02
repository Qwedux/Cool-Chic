# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from functools import cached_property
from typing import OrderedDict, Tuple, TypeAlias

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
from lossless.component.core.arm import (Arm, ArmParameter, _get_neighbor,
                                         _get_non_zero_pixel_ctx_index,
                                         _laplace_cdf)
from lossless.component.core.arm_image import ImageArm, ImageARMParameter
from lossless.component.core.proba_output import ProbabilityOutput
from lossless.component.core.quantizer import (
    POSSIBLE_QUANTIZATION_NOISE_TYPE, POSSIBLE_QUANTIZER_TYPE, quantize)
from lossless.component.core.synthesis import Synthesis
from lossless.component.core.upsampling import Upsampling
from lossless.component.types import DescriptorCoolChic, DescriptorNN
from lossless.nnquant.expgolomb import measure_expgolomb_rate
from lossless.util.device import PossibleDevice
from lossless.util.image import ImageSize
from torch import Tensor, nn
from typing_extensions import TypedDict, assert_never


@dataclass(frozen=True, kw_only=True)
class CoolChicEncoderParameter:
    layers_synthesis: Sequence[str]
    n_ft_per_res: Sequence[int]
    image_arm_parameters: ImageARMParameter
    arm_latent_parameters: ArmParameter
    encoder_gain: int
    ups_k_size: int
    ups_preconcat_k_size: int
    latent_freq_precision: int
    use_color_regression: bool
    img_size: ImageSize

    @cached_property
    def latent_n_grids(self) -> int:
        return len(self.n_ft_per_res)

    def pretty_string(self) -> str:
        ATTRIBUTE_WIDTH = 25
        VALUE_WIDTH = 80

        s = ""
        for k in fields(self):
            s += f"{k.name:<{ATTRIBUTE_WIDTH}}: {str(getattr(self, k.name)):<{VALUE_WIDTH}}\n"
        s += "\n"
        return s


@dataclass
class CoolChicEncoderOutput(TypedDict):
    # NOTE: This has to be a TypedDict for automatic measurement of MACs
    mu: Tensor
    scale: Tensor
    rate: Tensor
    latent_bpd: Tensor

@dataclass(frozen=True)
class UseImageARMOnly:
    pass

@dataclass(frozen=True)
class UseUntilSynthesis:
    pass

@dataclass(frozen=True)
class UseFullModel:
    pass

CoolChicComputingMode: TypeAlias = UseImageARMOnly | UseUntilSynthesis | UseFullModel



class CoolChicEncoder(nn.Module):
    """CoolChicEncoder for a single frame."""

    non_zero_pixel_ctx_index: Tensor

    def __init__(self, param: CoolChicEncoderParameter, computing_mode: CoolChicComputingMode, device: PossibleDevice):
        super().__init__()

        # Everything is stored inside param
        self.param = param
        self.computing_mode = computing_mode
        self.switch_computing_mode(computing_mode)
        self.device: PossibleDevice = device


        # ================== Synthesis related stuff ================= #
        # Encoder-side latent gain applied prior to quantization, one per feature
        self.encoder_gains = param.encoder_gain

        # Populate the successive grids
        self.size_per_latent = []
        self.latent_grids = nn.ParameterList()
        for i in range(self.param.latent_n_grids):
            h_grid, w_grid = int(math.ceil(self.param.img_size.height / (2**i))), int(math.ceil(self.param.img_size.width / (2**i)))
            c_grid = self.param.n_ft_per_res[i]
            cur_size = (1, c_grid, h_grid, w_grid)

            self.size_per_latent.append(cur_size)
            self.latent_grids.append(nn.Parameter(torch.empty(cur_size), requires_grad=True))
        self.initialize_latent_grids()

        # Instantiate the synthesis MLP with as many inputs as the number
        # of latent channels
        self.synthesis = Synthesis(
            sum([latent_size[1] for latent_size in self.size_per_latent]),
            self.param.layers_synthesis,
        )
        # ================== Synthesis related stuff ================= #

        # ===================== Upsampling stuff ===================== #
        self.upsampling = Upsampling(
            ups_k_size=self.param.ups_k_size,
            ups_preconcat_k_size=self.param.ups_preconcat_k_size,
            # Instantiate one different upsampling and pre-concatenation
            # filters for each of the upsampling step. Could also be set to one
            # to share the same filter across all latents.
            n_ups_kernel=self.param.latent_n_grids - 1,
            n_ups_preconcat_kernel=self.param.latent_n_grids - 1,
        )
        # ===================== Upsampling stuff ===================== #

        # ===================== ARM related stuff ==================== #
        # Create the probability model for the main INR. It uses a spatial context
        # parameterized by the spatial context

        # For a given mask size N (odd number e.g. 3, 5, 7), we have at most
        # (N * N - 1) / 2 context pixels in it.
        # Example, a 9x9 mask as below has 40 context pixel (indicated with 1s)
        # available to predict the pixel '*'
        #   1 1 1 1 1 1 1 1 1
        #   1 1 1 1 1 1 1 1 1
        #   1 1 1 1 1 1 1 1 1
        #   1 1 1 1 1 1 1 1 1
        #   1 1 1 1 * 0 0 0 0
        #   0 0 0 0 0 0 0 0 0
        #   0 0 0 0 0 0 0 0 0
        #   0 0 0 0 0 0 0 0 0
        #   0 0 0 0 0 0 0 0 0

        # No more than 40 context pixels i.e. a 9x9 mask size (see example above)
        max_mask_size = 9
        max_context_pixel = int((max_mask_size**2 - 1) / 2)
        assert self.param.image_arm_parameters.context_size <= max_context_pixel, (
            f"You can not have more context pixels "
            f" than {max_context_pixel}. Found {self.param.image_arm_parameters.context_size}"
        )

        # Mask of size 2N + 1 when we have N rows & columns of context.
        self.mask_size = max_mask_size

        # 1D tensor containing the indices of the selected context pixels.
        # register_buffer for automatic device management. We set persistent to false
        # to simply use the "automatically move to device" function, without
        # considering non_zero_pixel_ctx_index as a parameters (i.e. returned
        # by self.parameters())
        self.register_buffer(
            "non_zero_pixel_ctx_index",
            _get_non_zero_pixel_ctx_index(self.param.image_arm_parameters.context_size),
            persistent=False,
        )

        # ===================== ARM related stuff ==================== #
        self.arm = Arm(
            self.param.image_arm_parameters.context_size, self.param.image_arm_parameters.n_hidden_layers, self.param.image_arm_parameters.hidden_layer_dim
        )
        self.image_arm = ImageArm(self.param.image_arm_parameters)
        self.proba_output = ProbabilityOutput(self.param.use_color_regression)
        # self.modules_to_send = [tmp.name for tmp in fields(DescriptorCoolChic)]

        # ======================== Monitoring ======================== #
        # Pretty string representing the decoder complexity
        self.flops_str = ""
        # Total number of multiplications to decode the image
        self.total_flops = 0.0
        self.flops_per_module = {k: 0 for k in self.modules_to_send}
        # Fill the two attributes aboves
        # self.get_flops()
        # ======================== Monitoring ======================== #

        # Track the quantization step of each neural network, None if the
        # module is not yet quantized
        self.nn_q_step: Mapping[str, DescriptorNN] = {
            k: DescriptorNN(weight=None, bias=None) for k in self.modules_to_send
        }

        # Track the exponent of the exp-golomb code used for the NN parameters.
        # None if module is not yet quantized
        self.nn_expgol_cnt: Mapping[str, DescriptorNN] = {
            k: DescriptorNN(weight=None, bias=None) for k in self.modules_to_send
        }

        # Copy of the full precision parameters, set just before calling the
        # quantize_model() function. This is done through the
        # self._store_full_precision_param() function
        self.full_precision_param = None

    # ------- Actual forward
    def forward(
        self,
        image: Tensor = torch.empty(0),
        quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "kumaraswamy",
        quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
        soft_round_temperature: Tensor | None = torch.tensor(0.3),
        noise_parameter: Tensor | None = torch.tensor(1.0),
        AC_MAX_VAL: int = -1,
    ) -> CoolChicEncoderOutput:
        # ! Order of the operations are important as these are asynchronous
        # ! CUDA operations. Some ordering are faster than other...
        
        if isinstance(self.computing_mode, UseUntilSynthesis) or isinstance(self.computing_mode, UseFullModel):
            # ------ Encoder-side: quantize the latent
            # Convert the N [1, C, H_i, W_i] 4d latents with different resolutions
            # to a single flat vector. This allows to call the quantization
            # only once, which is faster
            encoder_side_flat_latent = torch.cat([latent_i.view(-1) for latent_i in self.latent_grids])

            flat_decoder_side_latent = quantize(
                encoder_side_flat_latent * self.encoder_gains,
                quantizer_noise_type if self.training else "none",
                quantizer_type if self.training else "hardround",
                soft_round_temperature,
                noise_parameter,
            )

            # Clamp latent if we need to write a bitstream
            if AC_MAX_VAL != -1:
                flat_decoder_side_latent = torch.clamp(
                    flat_decoder_side_latent, -AC_MAX_VAL, AC_MAX_VAL + 1
                )

            # Convert back the 1d tensor to a list of N [1, C, H_i, W_i] 4d latents.
            # This require a few additional information about each individual
            # latent dimension, stored in self.size_per_latent
            decoder_side_latent = []
            cnt = 0
            for latent_size in self.size_per_latent:
                b, c, h, w = latent_size  # b should be one
                latent_numel = b * c * h * w
                decoder_side_latent.append(
                    flat_decoder_side_latent[cnt : cnt + latent_numel].view(latent_size)
                )
                cnt += latent_numel

            # ----- ARM to estimate the distribution and the rate of each latent
            # As for the quantization, we flatten all the latent and their context
            # so that the ARM network is only called once.
            # flat_latent: [N, 1] tensor describing N latents
            # flat_context: [N, context_size] tensor describing each latent context

            # Get all the context as a single 2D vector of size [B, context size]
            latent_context_flat = torch.cat(
                [
                    _get_neighbor(
                        spatial_latent_i,
                        self.mask_size,
                        self.non_zero_pixel_ctx_index,
                    )
                    for spatial_latent_i in decoder_side_latent
                ],
                dim=0,
            )

            # Get all the B latent variables as a single one dimensional vector
            flat_latent = torch.cat(
                [spatial_latent_i.view(-1) for spatial_latent_i in decoder_side_latent],
                dim=0,
            )

            # Feed the spatial context to the arm MLP and get mu and scale
            flat_mu, flat_scale, _ = self.arm(latent_context_flat)

            # Compute the rate (i.e. the entropy of flat latent knowing mu and scale)
            proba = torch.clamp_min(
                _laplace_cdf(flat_latent + 0.5, flat_mu, flat_scale)
                - _laplace_cdf(flat_latent - 0.5, flat_mu, flat_scale),
                min=2**-16,  # No value can cost more than 16 bits.
            )
            flat_rate = -torch.log2(proba)

            # Upsampling and synthesis to get the output
            ups_out = self.upsampling(decoder_side_latent)
            # has e.g. shape [1, 9, H, W]
            raw_synth_out = self.synthesis(ups_out)
            if isinstance(self.computing_mode, UseFullModel):
                raw_synth_out = self.image_arm(image, raw_synth_out)
            mu, scale = self.proba_output(raw_synth_out, image)            
        elif isinstance(self.computing_mode, UseImageARMOnly):
            flat_rate = torch.zeros(
                sum(map(lambda x: x[0] * x[1] * x[2] * x[3], self.size_per_latent)), device=self.device.materialize()
            )
            raw_synth_out = torch.zeros(image.shape[0], 9 if self.param.use_color_regression else 6, image.shape[2], image.shape[3], device=self.device.materialize())
            raw_synth_out = self.image_arm(image, raw_synth_out)
            mu, scale = self.proba_output(raw_synth_out, image)
        else:
            raise ValueError(f"Unknown computing mode: {self.computing_mode}")
        
        res: CoolChicEncoderOutput = CoolChicEncoderOutput(
            mu=mu,
            scale=scale,
            rate=flat_rate,
            latent_bpd=flat_rate.sum() / self.param.img_size.height / self.param.img_size.width / 3,
        )

        return res

    def get_latents_raw_synth_out(
        self,
        quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "kumaraswamy",
        quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
        soft_round_temperature: Tensor | None = torch.tensor(0.3),
        noise_parameter: Tensor | None = torch.tensor(1.0),
        AC_MAX_VAL: int = -1,
    ):
        # ! Order of the operations are important as these are asynchronous
        # ! CUDA operations. Some ordering are faster than other...

        # ------ Encoder-side: quantize the latent
        # Convert the N [1, C, H_i, W_i] 4d latents with different resolutions
        # to a single flat vector. This allows to call the quantization
        # only once, which is faster
        encoder_side_flat_latent = torch.cat([latent_i.view(-1) for latent_i in self.latent_grids])

        flat_decoder_side_latent = quantize(
            encoder_side_flat_latent * self.encoder_gains,
            quantizer_noise_type if self.training else "none",
            quantizer_type if self.training else "hardround",
            soft_round_temperature,
            noise_parameter,
        )

        # Clamp latent if we need to write a bitstream
        if AC_MAX_VAL != -1:
            flat_decoder_side_latent = torch.clamp(
                flat_decoder_side_latent, -AC_MAX_VAL, AC_MAX_VAL + 1
            )

        # Convert back the 1d tensor to a list of N [1, C, H_i, W_i] 4d latents.
        # This require a few additional information about each individual
        # latent dimension, stored in self.size_per_latent
        decoder_side_latent = []
        cnt = 0
        for latent_size in self.size_per_latent:
            b, c, h, w = latent_size  # b should be one
            latent_numel = b * c * h * w
            decoder_side_latent.append(
                flat_decoder_side_latent[cnt : cnt + latent_numel].view(latent_size)
            )
            cnt += latent_numel

        # Upsampling and synthesis to get the output
        ups_out = self.upsampling(decoder_side_latent)
        # has e.g. shape [1, 9, H, W]
        raw_synth_out = self.synthesis(ups_out)
        return raw_synth_out, decoder_side_latent

    # ------- Getter / Setter and Initializer
    def get_param(self) -> OrderedDict[str, Tensor]:
        param = OrderedDict({})
        param.update(
            {
                # Detach & clone to create a copy
                f"latent_grids.{k}": v.detach().clone()
                for k, v in self.latent_grids.named_parameters()
            }
        )
        param.update({f"arm.{k}": v for k, v in self.arm.get_param().items()})
        # I broke the following as image_arms is now a list of modules
        param.update(
            {f"image_arm.{k}": v.detach().clone() for k, v in self.image_arm.named_parameters()}
        )
        param.update({f"upsampling.{k}": v for k, v in self.upsampling.get_param().items()})
        param.update({f"synthesis.{k}": v for k, v in self.synthesis.get_param().items()})
        return param

    def set_param(self, param: OrderedDict[str, Tensor]):
        self.load_state_dict(param)

    def initialize_latent_grids(self) -> None:
        for latent_index, latent_value in enumerate(self.latent_grids):
            self.latent_grids[latent_index] = nn.Parameter(
                torch.zeros_like(latent_value), requires_grad=True
            )

    def reinitialize_parameters(self):
        self.arm.reinitialize_parameters()
        self.upsampling.reinitialize_parameters()
        self.synthesis.reinitialize_parameters()
        self.initialize_latent_grids()

        # Reset the quantization steps and exp-golomb count of the neural
        # network to None since we are resetting the parameters.
        self.nn_q_step = {
            k: DescriptorNN(weight=None, bias=None) for k in self.modules_to_send
        }
        self.nn_expgol_cnt = {
            k: DescriptorNN(weight=None, bias=None) for k in self.modules_to_send
        }

    def _store_full_precision_param(self) -> None:
        if self.full_precision_param is not None:
            print(
                "Warning: overwriting already saved full-precision parameters"
                " in CoolChicEncoder _store_full_precision_param()."
            )
        no_q_step = True
        for _, q_step_dict in self.nn_q_step.items():
            for _, q_step in q_step_dict.items(): # type: ignore
                if q_step is not None:
                    no_q_step = False
        assert no_q_step, (
            "Trying to store full precision parameters, while CoolChicEncoder "
            "nn_q_step attributes is not full of None. This means that the "
            "parameters have already been quantized... aborting!"
        )

        no_expgol_cnt = True
        for _, expgol_cnt_dict in self.nn_expgol_cnt.items():
            for _, expgol_cnt in expgol_cnt_dict.items(): # type: ignore
                if expgol_cnt is not None:
                    no_expgol_cnt = False
        assert no_expgol_cnt, (
            "Trying to store full precision parameters, while CoolChicEncoder "
            "nn_expgol_cnt attributes is not full of None. This means that the "
            "parameters have already been quantized... aborting!"
        )

        # All good, simply save the parameters
        self.full_precision_param = self.get_param()

    def _load_full_precision_param(self) -> None:
        assert self.full_precision_param is not None, (
            "Trying to load full precision parameters but " "self.full_precision_param is None"
        )

        self.set_param(self.full_precision_param)

        # Reset the side information about the quantization step and expgol cnt
        # so that the rate is no longer computed by the test() function.
        self.nn_q_step = {
            k: DescriptorNN(weight=None, bias=None) for k in self.modules_to_send
        }

        self.nn_expgol_cnt = {
            k: DescriptorNN(weight=None, bias=None) for k in self.modules_to_send
        }

    # ------- Get flops, neural network rates and quantization step
    def get_flops(self) -> None:
        self = self.train(mode=False)
        assert self.param.img_size is not None

        flops = FlopCountAnalysis(
            self,
            (
                torch.empty(1, 3, self.param.img_size.height, self.param.img_size.width, device=self.device.materialize()),  # image
                "none",  # Quantization noise
                "hardround",  # Quantizer type
                0.3,  # Soft round temperature
                0.1,  # Noise parameter
                -1,  # AC_MAX_VAL
            ),  # type: ignore
        )
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)

        self.total_flops = flops.total()
        for k in self.flops_per_module:
            self.flops_per_module[k] = flops.by_module()[k]

        self.flops_str = flop_count_table(flops, max_depth=4)
        del flops

        self = self.train(mode=True)

    def get_network_rate(self) -> Tuple[DescriptorCoolChic, float]:
        rate_per_module: DescriptorCoolChic = {
            module_name: DescriptorNN(weight=0.0, bias=0.0) for module_name in self.modules_to_send
        } # type: ignore

        total_rate = 0.0

        for module_name in self.modules_to_send:
            cur_module = getattr(self, module_name)
            rate_per_module[module_name] = measure_expgolomb_rate( # type: ignore
                cur_module,
                self.nn_q_step.get(module_name), # type: ignore
                self.nn_expgol_cnt.get(module_name), # type: ignore
            ) # type: ignore

            total_rate += sum(rate_per_module[module_name].values()) # type: ignore

        return rate_per_module, total_rate

    def get_network_quantization_step(self) -> DescriptorCoolChic:
        return self.nn_q_step # type: ignore

    def get_network_expgol_count(self) -> DescriptorCoolChic:
        return self.nn_expgol_cnt # type: ignore

    def str_complexity(self) -> str:
        if not self.flops_str:
            self.get_flops()

        msg_total_mac = "----------------------------------\n"
        msg_total_mac += f"Total MAC / decoded pixel: {self.get_total_mac_per_pixel():.1f}"
        msg_total_mac += "\n----------------------------------"

        return self.flops_str + "\n\n" + msg_total_mac

    def get_total_mac_per_pixel(self) -> float:       
        if not self.flops_str:
            self.get_flops()

        assert self.param.img_size is not None, "Image size is not set"
        n_pixels = self.param.img_size.height * self.param.img_size.width
        return self.total_flops / n_pixels

    # ------- Useful functions
    def to_device(self, device: PossibleDevice) -> None:
        self.device = device
        self = self.to(device.materialize())

        # Push integerized weights and biases of the mlp (resp qw and qb) to
        # the required device
        for idx_layer, layer in enumerate(self.arm.mlp):
            if hasattr(layer, "qw"):
                if layer.qw is not None:
                    self.arm.mlp[idx_layer].qw = layer.qw.to(self.device.materialize())

            if hasattr(layer, "qb"):
                if layer.qb is not None:
                    self.arm.mlp[idx_layer].qb = layer.qb.to(self.device.materialize())

        self.image_arm = self.image_arm.to(self.device.materialize())
        self.image_arm.non_zero_image_arm_ctx_index = (
            self.image_arm.non_zero_image_arm_ctx_index.to(self.device.materialize())
        )
        for expert in self.image_arm.image_arm_models:
            assert isinstance(expert, nn.ModuleList)
            for model in expert:
                assert isinstance(model, torch.nn.Sequential)
                for idx_layer, layer in enumerate(model):
                    layer.to(self.device.materialize())
                    if hasattr(layer, "qw"):
                        if layer.qw is not None:
                            model[idx_layer].qw = layer.qw.to(self.device.materialize())

                    if hasattr(layer, "qb"):
                        if layer.qb is not None:
                            model[idx_layer].qb = layer.qb.to(self.device.materialize())


    def load_from_disk(self, path: str) -> None:
        state_dict = torch.load(path, map_location="cpu")
        self.load_state_dict(state_dict)
    
    def save_to_disk(self, path: str) -> None:
        torch.save(self.state_dict(), path)
    
    def switch_computing_mode(self, computing_mode: CoolChicComputingMode) -> None:
        self.computing_mode = computing_mode
        print(f"Switching to computing mode: {self.computing_mode}")
        match self.computing_mode:
            case UseImageARMOnly():
                self.modules_to_send = ["image_arm"]
            case UseUntilSynthesis():
                self.modules_to_send = ["arm", "upsampling", "synthesis"]
            case UseFullModel():
                self.modules_to_send = ["arm", "upsampling", "synthesis", "image_arm"]
            case _:
                assert_never(self.computing_mode)
        print(f"Updataed Modules to send: {self.modules_to_send}")