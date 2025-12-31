# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import copy
import time
import typing
from dataclasses import dataclass
from typing import TypeGuard, get_args

import torch
from lossless.component.coolchic import CoolChicEncoder
from lossless.training.loss import LossFunctionOutput
from lossless.training.manager import ImageEncoderManager
from lossless.training.test import test
from lossless.training.train_phase import _train_single_phase
from lossless.util.color_transform import ColorBitdepths
from lossless.util.device import POSSIBLE_DEVICE
from lossless.util.logger import TrainingLogger
from lossless.util.misc import mem_info


def is_possible_device(val: str) -> TypeGuard[POSSIBLE_DEVICE]:
    return val in get_args(POSSIBLE_DEVICE)

@dataclass
class WarmupCandidate:
    metrics: LossFunctionOutput | None
    id: int
    encoder: CoolChicEncoder


def warmup(
    image_encoder_manager: ImageEncoderManager,
    template_model: CoolChicEncoder,
    target_image: torch.Tensor,
    logger: TrainingLogger,
) -> CoolChicEncoder:
    """Perform the warm-up for a frame encoder. It consists in multiple stages
    with several candidates, filtering out the best N candidates at each stage.
    For instance, we can start with 8 different FrameEncoder. We train each of
    them for 400 iterations. Then we keep the best 4 of them for 400 additional
    iterations, while finally keeping the final best one.

    .. warning::

        The parameter ``frame_encoder_manager`` tracking the encoding time of
        the frame (``total_training_time_sec``) and the number of encoding
        iterations (``iterations_counter``) is modified** in place** by this
        function.

    Args:
        frame_encoder_manager: Contains (among other things) the rate
            constraint :math:`\\lambda` and description of the
            warm-up preset. It is also used to track the total encoding time
            and encoding iterations. Modified in place.
        list_candidates: The different candidates among which the warm-up will
            find the best starting point.
        frame: The original image to be compressed and its references.
        device: On which device should the training run.

    Returns:
        Warmuped frame encoder, with a great initialization.
    """

    start_time = time.time()
    warmup = image_encoder_manager.preset.warmup
    if len(warmup.phases) == 0:
        logger.log_result("No warm-up phase defined, skipping warm-up.")
        return template_model

    num_starting_candidates = warmup.phases[0].candidates
    _col_width = 14

    # Construct the list of candidates. Each of them has its own parameters,
    # unique ID and metrics (not yet evaluated so it is set to None).
    all_candidates: list[WarmupCandidate] = [
        WarmupCandidate(
            metrics=None, id=id_candidate, encoder=CoolChicEncoder(template_model.param)
        )
        for id_candidate in range(num_starting_candidates)
    ]

    for idx_warmup_phase, warmup_phase in enumerate(warmup.phases):
        logger.log_result(f'{"-" * 30}  Warm-up phase: {idx_warmup_phase:>2} {"-" * 30}')

        mem_info(f"Warmup-{idx_warmup_phase:02d}")

        # At the beginning of the all warm-up phases except the first one,
        # keep the desired number of best candidates.
        if idx_warmup_phase != 0:
            n_elements_to_remove = len(all_candidates) - warmup_phase.candidates
            for _ in range(n_elements_to_remove):
                all_candidates.pop()

        # # Check that we do have different candidates with different parameters
        # print('------\nbefore')
        # for x in all_candidates:
        #     print(f"{x.id}   {sum([v.abs().sum() for k, v in x.encoder.get_param().items() if 'synthesis' in k])}")

        # Train all (remaining) candidates for a little bit
        for i in range(warmup_phase.candidates):
            cur_candidate_model = all_candidates[i]
            cur_id = cur_candidate_model.id

            logger.log_result(f"\nCandidate n° {i:<2}, ID = {cur_id:<2}:" + "\n-------------------------\n")
            logger.log_result(mem_info(f"Warmup-cand-in {idx_warmup_phase:02d}-{i:02d}"))

            if is_possible_device(template_model.device):
                template_device = template_model.device
            else:
                raise ValueError(f"Invalid device: {template_model.device}")
            cur_candidate_model.encoder.to_device(template_device)
            initial_encoder_logs = test(
                cur_candidate_model.encoder,
                target_image,
                image_encoder_manager,
            )

            cur_candidate_model.encoder, _ = _train_single_phase(
                model=cur_candidate_model.encoder,
                target_image=target_image,
                image_encoder_manager=image_encoder_manager,
                training_phase=warmup_phase.training_phase,
                logger=logger,
                encoder_logs_best=initial_encoder_logs,
            )

            cur_candidate_model.metrics = test(
                cur_candidate_model.encoder,
                target_image,
                image_encoder_manager,
            )

            # Put the updated candidate back into the list.
            all_candidates[i] = cur_candidate_model

        all_candidates = sorted(
            all_candidates, key=lambda x: x.metrics.loss if x.metrics is not None else float("inf")
        )

        # # Check that we do have different candidates with different parameters
        # for x in all_candidates:
        #     print(f"{x.id}   {sum([v.abs().sum() for k, v in x.encoder.get_param().items() if 'synthesis' in k])}")
        # print('after\n------')

        # Print the results of this warm-up phase
        s = "\n\nPerformance at the end of the warm-up phase:\n\n"
        s += f'{"ID":^{6}}|{"loss":^{_col_width}}|{"img_bpd":^{_col_width}}|{"latent_bpd":^{_col_width}}|\n'
        s += f'------|{"-" * _col_width}|{"-" * _col_width}|{"-" * _col_width}|\n'
        for candidate in all_candidates:
            assert (
                candidate.metrics is not None
            ), "Metrics should have been evaluated for all candidates."
            s += f"{candidate.id:^{6}}|"
            s += f"{candidate.metrics.loss.item():^{_col_width}.4f}|"
            s += f"{candidate.metrics.rate_img_bpd:^{_col_width}.4f}|"
            s += f"{candidate.metrics.rate_latent_bpd:^{_col_width}.4f}|"
            s += "\n"
        logger.log_result(s)

    # Keep only the best model
    frame_encoder = copy.deepcopy(all_candidates[0].encoder)

    # We've already worked for that many second during warm up
    warmup_duration = time.time() - start_time

    logger.log_result("Intra Warm-up is done!")
    logger.log_result(f"Intra Warm-up time [s]: {warmup_duration:.2f}")
    logger.log_result(f"Intra Winner ID       : {all_candidates[0].id}\n")

    return frame_encoder
