# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

from __future__ import annotations

import gc
import time
from typing import cast

import torch
from lossless.component.coolchic import CoolChicEncoder
from lossless.component.core.arm_image import MultiImageArmDescriptor
from lossless.training.manager import ImageEncoderManager
from lossless.training.test import test
from lossless.training.train_phase import _train_single_phase
from lossless.training.warmup import warmup
from lossless.util.logger import TrainingLogger


def train(
    model: CoolChicEncoder,
    target_image: torch.Tensor,
    image_encoder_manager: ImageEncoderManager,
    logger: TrainingLogger,
) -> CoolChicEncoder:
    start_time = time.time()
    logger.log_result(f"Starting warmup with {model.image_arm.image_arm_models}")
    logger.log_result(f"Image ARM setup: {cast(MultiImageArmDescriptor, model.image_arm.params.multi_region_image_arm_specification).num_experts}")
    model = warmup(
        image_encoder_manager=image_encoder_manager,
        template_model=model,
        target_image=target_image,
        logger=logger,
    )
    # clear torch cache
    torch.cuda.empty_cache()
    gc.collect()
    model.image_arm.params = model.image_arm.params.make_new_image_arm_specification(
        num_parts_per_col=image_encoder_manager.multi_region_image_arm_setup[0],
        num_parts_per_row=image_encoder_manager.multi_region_image_arm_setup[1],
    )
    model.image_arm.reinitialize_image_arm_experts(
        num_experts=cast(MultiImageArmDescriptor, model.image_arm.params.multi_region_image_arm_specification).num_experts,
        pretrained_expert_index=0,
    )
    logger.log_result(f"Image ARM setup: {cast(MultiImageArmDescriptor, model.image_arm.params.multi_region_image_arm_specification).num_experts}")
    logger.log_result(f"Image ARM setup: {cast(MultiImageArmDescriptor, model.image_arm.params.multi_region_image_arm_specification).routing_grid}")

    initial_encoder_logs = test(
        model, target_image, image_encoder_manager
    )
    encoder_logs_best = initial_encoder_logs
    
    for training_phase in image_encoder_manager.preset.training_phases:
        logger.log_training(f"Starting new training phase:\n{training_phase.pretty_string()}")
        model, encoder_logs_best = _train_single_phase(
            model=model,
            target_image=target_image,
            image_encoder_manager=image_encoder_manager,
            training_phase=training_phase,
            logger=logger,
            encoder_logs_best=encoder_logs_best,
        )
    image_encoder_manager.total_training_time_sec += time.time() - start_time

    encoder_logs = test(
        model,
        target_image,
        image_encoder_manager,
    )
    logger.log_result(f"At the end of the training: " + str(encoder_logs))
    return model
