# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md



import time
from typing import List, Tuple

import torch
from lossless.component.coolchic import CoolChicEncoder
from lossless.training.manager import ImageEncoderManager
from lossless.training.test import test
from lossless.training.train_phase import _train_single_phase
from lossless.training.warmup import warmup
from lossless.util.color_transform import ColorBitdepths
from lossless.util.logger import TrainingLogger


def train(
    model: CoolChicEncoder,
    target_image: torch.Tensor,
    image_encoder_manager: ImageEncoderManager,
    logger: TrainingLogger,
    color_bitdepths: ColorBitdepths,
) -> CoolChicEncoder:
    """Train a ``CoolChicEncoder`` and return the updated module. This function is
    supposed to be called any time we want to optimize the parameters of a
    CoolChicEncoder, either during the warm-up (competition of multiple possible
    initializations) or during of the stages of the actual training phase.

    The module is optimized according to the following loss function:

    .. math::

        \\mathcal{L} = ||\\mathbf{x} - \\hat{\\mathbf{x}}||^2 + \\lambda
        \\mathrm{R}(\\hat{\\mathbf{x}}), \\text{ with } \\begin{cases}
            \\mathbf{x} & \\text{the original image}\\\\ \\hat{\\mathbf{x}} &
            \\text{the coded image}\\\\ \\mathrm{R}(\\hat{\\mathbf{x}}) &
            \\text{A measure of the rate of } \\hat{\\mathbf{x}}
        \\end{cases}

    .. warning::

        The parameter ``image_encoder_manager`` tracking the encoding time of
        the frame (``total_training_time_sec``) and the number of encoding
        iterations (``iterations_counter``) is modified **in place** by this
        function.

    Args:
        image_encoder_manager: Module to be trained.
        frame: The original image to be compressed and its references.
        image_encoder_manager: Contains (among other things) the rate
            constraint :math:`\\lambda`. It is also used to track the total
            encoding time and encoding iterations. Modified in place.
        start_lr: Initial learning rate. Either constant for the entire
            training or schedule using a cosine scheduling, see below for more
            details. Defaults to 1e-2.
        cosine_scheduling_lr: True to schedule the learning
            rate from ``start_lr`` at iteration n°0 to 0 at iteration
            n° ``max_iterations``. Defaults to True.
        max_iterations: Do at most ``max_iterations`` iterations.
            The actual number of iterations can be made smaller through the
            patience mechanism. Defaults to 10000.
        frequency_validation: Check (and print) the performance
            each ``frequency_validation`` iterations. This drives the patience
            mechanism. Defaults to 100.
        patience: After ``patience`` iterations without any
            improvement to the results, exit the training. Patience is disabled
            by setting ``patience = max_iterations``. If patience is used alongside
            cosine_scheduling_lr, then it does not end the training. Instead,
            we simply reload the best model so far once we reach the patience,
            and the training continue. Defaults to 10.
        optimized_module: List of modules to be optimized. Most often you'd
            want to use ``optimized_module = ['all']``. Defaults to ``['all']``.
        quantizer_type: What quantizer to
            use during training. See :doc:`encoder/component/core/quantizer.py
            <../component/core/quantizer>` for more information. Defaults to
            ``"softround"``.
        quantizer_noise_type: The random noise used by the quantizer. More
            information available in
            :doc:`encoder/component/core/quantizer.py
            <../component/core/quantizer>`. Defaults to ``"kumaraswamy"``.
        softround_temperature: The softround temperature is linearly scheduled
            during the training. At iteration n° 0 it is equal to
            ``softround_temperature[0]`` while at iteration n° ``max_itr`` it is
            equal to ``softround_temperature[1]``. Note that the patience might
            interrupt the training before it reaches this last value.
            Defaults to (0.3, 0.2).
        noise_parameter: The random noise temperature is linearly scheduled
            during the training. At iteration n° 0 it is equal to
            ``noise_parameter[0]`` while at iteration n° ``max_itr`` it is equal
            to ``noise_parameter[1]``. Note that the patience might interrupt
            the training before it reaches this last value. Defaults to (2.0,
            1.0).

    Returns:
        The trained frame encoder.
    """
    start_time = time.time()

    # ------ Keep track of the best loss and model
    # Perform a first test to get the current best logs (it includes the loss)

    # do warmups hare

    model = warmup(
        image_encoder_manager=image_encoder_manager,
        template_model=model,
        target_image=target_image,
        logger=logger,
        color_bitdepths=color_bitdepths,
    )
    initial_encoder_logs = test(
        model, target_image, image_encoder_manager, color_bitdepths=color_bitdepths
    )
    encoder_logs_best = initial_encoder_logs

    for training_phase in image_encoder_manager.preset.training_phases:
        logger.log_training(f"Starting new training phase:\n{training_phase.pretty_string()}")
        model = _train_single_phase(
            model=model,
            target_image=target_image,
            image_encoder_manager=image_encoder_manager,
            training_phase=training_phase,
            logger=logger,
            color_bitdepths=color_bitdepths,
            encoder_logs_best=encoder_logs_best,
        )
    image_encoder_manager.total_training_time_sec += time.time() - start_time

    encoder_logs = test(
        model,
        target_image,
        image_encoder_manager,
        color_bitdepths=color_bitdepths,
    )
    logger.log_result(f"At the end of the training: " + str(encoder_logs))
    return model
