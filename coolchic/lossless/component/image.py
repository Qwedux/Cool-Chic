# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import copy
import os
import time
from enum import Enum
from typing import Dict, List

import torch
from lossless.component.coolchic import CoolChicEncoder, CoolChicEncoderParameter
from lossless.component.frame import (
    NAME_COOLCHIC_ENC,
    FrameEncoder,
    load_frame_encoder,
)

from enc.io.format.png import write_png
from enc.io.format.ppm import write_ppm
from enc.io.format.yuv import write_yuv
from enc.io.framedata import FrameData
from enc.io.io import load_frame_data_from_file
from enc.nnquant.quantizemodel import quantize_model
from lossless.training.manager import FrameEncoderManager
from lossless.training.test import test
from lossless.training.train import train
from lossless.training.warmup import warmup
from enc.utils.codingstructure import CodingStructure, Frame
from enc.utils.device import POSSIBLE_DEVICE
from enc.utils.misc import mem_info
from torch import Tensor


# =========================== Job management ============================ #
# Exiting the program with 42 will requeue the job
class TrainingExitCode(Enum):
    END = 0
    REQUEUE = 42


def _is_job_over(start_time: float, max_duration_job_min: int = 45) -> bool:
    """Return true if current time is more than max_duration_job_min after start time.
    Use -1 for max_job_duration_min to always return False


    Args:
        start_time (float): time.time() at the start of the program.
        max_duration_job_min (int): How long we should run. If -1, we never stop
            i.e. we always return False.

    Returns:
        bool: True if current time is more than max_duration_job_min after start time.
    """
    if max_duration_job_min < 0:
        return False

    return (time.time() - start_time) / 60 >= max_duration_job_min


# =========================== Job management ============================ #


def encode_one_frame(
    video_path: str,
    coding_structure: CodingStructure,
    coolchic_enc_param: Dict[NAME_COOLCHIC_ENC, CoolChicEncoderParameter],
    frame_encoder_manager: FrameEncoderManager,
    coding_index: int,
    job_duration_min: int = -1,
    device: POSSIBLE_DEVICE = "cpu",
    print_detailed_archi: bool = False,
) -> TrainingExitCode:
    start_time = time.time()
    frame = coding_structure.get_frame_from_coding_order(coding_index)
    assert frame is not None

    def change_n_out_synth(layers_synth: List[str], n_out: int) -> List[str]:
        """Change the number of output features in the list of strings
        describing the synthesis architecture. It replaces "X" with n_out. E.g.

        From [8-1-linear-relu,X-1-linear-none,X-3-residual-none]
        To   [8-1-linear-relu,2-1-linear-none,2-3-residual-none]

        If n_out = 2

        Args:
            layers_synth (List[str]): List of strings describing the different
                synthesis layers
            n_out (int): Number of desired output.

        Returns:
            List[str]: List of strings with the proper number of output features.
        """
        return [lay.replace("X", str(n_out)) for lay in layers_synth]

    print(
        "Frame being encoded\n"
        "-------------------\n\n"
        f"{frame.pretty_string(show_header=True, show_bottom_line=True)}"
    )

    if frame_encoder_manager.loop_counter >= frame_encoder_manager.n_loops:
        # This is the next frame to code
        next_frame = coding_structure.get_frame_from_coding_order(
            frame.coding_order + 1
        )
        msg = (
            f"Frame {frame.frame_type}{frame.display_order} has already "
            f"undergone {frame_encoder_manager.loop_counter} / "
            f"{frame_encoder_manager.n_loops} training loop(s).\n"
        )
        if next_frame is not None:
            msg += (
                "Hint: you can now encode frame "
                f"{next_frame.frame_type}{next_frame.display_order} "
                f"using --coding_idx={next_frame.coding_order}"
            )
        print(msg)

    torch.set_float32_matmul_precision("high")

    for index_loop in range(frame_encoder_manager.loop_counter, frame_encoder_manager.n_loops):
        # Load the original data and its references
        frame.set_frame_data(
            load_frame_data_from_file(
                video_path, frame.display_order + frame.frame_offset
            )
        )
        frame.set_refs_data(get_ref_data(frame, coding_structure))
        prefix_save = f"{_get_frame_path_prefix(frame.display_order)}"

        for cc_enc_name in coolchic_enc_param:
            assert frame.data is not None
            coolchic_enc_param[cc_enc_name].set_image_size(frame.data.img_size)

        match frame.frame_type:
            case "I":
                coolchic_enc_param["residue"].encoder_gain = 16
                coolchic_enc_param["residue"].layers_synthesis = (
                    change_n_out_synth(
                        coolchic_enc_param["residue"].layers_synthesis, 3
                    )
                )

            case _:
                raise ValueError(f"Unknown frame type {frame.frame_type}.")

        print(
            "-" * 80
            + "\n"
            + f'{" " * 30} Training loop {frame_encoder_manager.loop_counter + 1} / '
            + f"{frame_encoder_manager.n_loops}\n"
            + "-" * 80
        )

        # First loop, print some stuff!
        if frame_encoder_manager.loop_counter == 0:
            # Log a few details about the model
            print(f"\n{frame_encoder_manager.pretty_string()}")
            print(f"{frame_encoder_manager.preset.pretty_string()}")

            tmp_str = "Decoder architectures\n"
            tmp_str += "---------------------\n\n"
            for name, cc_enc_param in coolchic_enc_param.items():
                title = (
                    f"{name} decoder parameters\n"
                    f"{'-' * len(name)}-------------------\n"
                )
                tmp_str += title
                tmp_str += cc_enc_param.pretty_string() + "\n"

            print(tmp_str)

        frame = frame_to_device(frame, device)

        # Get the number of candidates from the initial warm-up phase
        n_candidates = frame_encoder_manager.preset.warmup.phases[0].candidates

        list_candidates = []
        for idx_candidate in range(n_candidates):
            assert frame.data is not None
            cur_frame_encoder = FrameEncoder(
                coolchic_enc_param=coolchic_enc_param,
                frame_type=frame.frame_type,
                frame_data_type=frame.data.frame_data_type,
                bitdepth=frame.data.bitdepth,
                index_references=frame.index_references,
                frame_display_index=frame.display_order,
            )

            list_candidates.append(cur_frame_encoder)

        # Use the first candidate of the list to log the architecture
        with open(f"{prefix_save}archi.txt", "w") as f_out:
            for cc_name, cc_enc in list_candidates[0].coolchic_enc.items():
                f_out.write(cc_name + "\n\n" + str(cc_enc) + "\n\n")
                f_out.write(cc_enc.str_complexity() + "\n")

        print(
            list_candidates[0].pretty_string(
                print_detailed_archi=print_detailed_archi
            )
            + "\n\n"
        )

        # Use warm-up to find the best initialization among the list
        # of candidates parameters.
        frame_encoder = warmup(
            frame_encoder_manager=frame_encoder_manager,
            list_candidates=list_candidates,
            frame=frame,
            device=device,
        )

        frame_encoder.to_device(device)

        # Compile only after the warm-up to compile only once.
        # No compilation for torch version anterior to 2.5.0
        major, minor = [int(x) for x in torch.__version__.split(".")[:2]]
        use_compile = False
        if major > 2:
            use_compile = True
        elif major == 2:
            use_compile = minor >= 5

        cur_preset = frame_encoder_manager.preset

        if cur_preset.preset_name == "debug":
            print("Skip compilation when debugging\n")
        elif (
            cur_preset._get_total_training_iterations(
                cur_preset.training_phases
            )
            <= 1000
        ):
            print("No compilation when training has less than 1000 iterations")
        elif not use_compile:
            print("No compilation for torch version anterior to 2.5.0\n")
        else:
            print("Compiling frame encoder!\n")
            torch._dynamo.reset()
            assert frame.data is not None
            frame_encoder = torch.compile(
                frame_encoder,
                dynamic=False,
                mode="reduce-overhead",
                # Some part of the frame_encoder forward (420-related stuff)
                # are not (yet) compatible with compilation. So we can't
                # capture the full graph for yuv420 frame
                fullgraph=frame.data.frame_data_type != "yuv420",
            )

        for idx_phase, training_phase in enumerate(frame_encoder_manager.preset.training_phases):
            print(f'{"-" * 30} Training phase: {idx_phase:>2} {"-" * 30}\n')
            mem_info("Training phase " + str(idx_phase))
            frame_encoder = train(
                model=frame_encoder,
                target_image=frame,
                frame_encoder_manager=frame_encoder_manager,
                start_lr=training_phase.lr,
                lmbda=frame_encoder_manager.lmbda,
                cosine_scheduling_lr=training_phase.schedule_lr,
                max_iterations=training_phase.max_itr,
                frequency_validation=training_phase.freq_valid,
                patience=training_phase.patience,
                optimized_module=training_phase.optimized_module,
                quantizer_type=training_phase.quantizer_type,
                quantizer_noise_type=training_phase.quantizer_noise_type,
                softround_temperature=training_phase.softround_temperature,
                noise_parameter=training_phase.noise_parameter,
            )

            if training_phase.quantize_model:
                # Store full precision parameters inside the
                # frame_encoder for later use if needed
                frame_encoder._store_full_precision_param()
                frame_encoder = quantize_model(
                    frame_encoder,
                    frame,
                    frame_encoder_manager,
                )

            phase_results = test(
                frame_encoder,
                frame,
                frame_encoder_manager,
            )

            print("\nResults at the end of the phase:")
            print("--------------------------------")
            print(
                f'\n{phase_results.pretty_string(show_col_name=True, mode="short")}\n'
            )

        # At the end of each loop, compute the final loss
        # assert isinstance(frame_encoder, FrameEncoder), (
        #     f"frame_encoder should be an instance of FrameEncoder, "
        #     f"found {type(frame_encoder)}"
        # )
        loop_results = test(
            frame_encoder,
            frame,
            frame_encoder_manager,
        )

        # We only care for the best_results.tsv
        # Write results file
        path_results_log = f"{prefix_save}results_loop_{frame_encoder_manager.loop_counter + 1}.tsv"
        with open(path_results_log, "w") as f_out:
            f_out.write(
                loop_results.pretty_string(show_col_name=True, mode="all")
                + "\n"
            )

        path_best_frame_enc = f"{prefix_save}frame_encoder.pt"

        # We've beaten our record, save the corresponding .pt + the decoded image
        frame_encoder_manager.loop_counter += 1
        if frame_encoder_manager.record_beaten(loop_results.loss):
            print(
                f"Best loss beaten at loop {frame_encoder_manager.loop_counter}"  # no need for +1 anymore, it's done above
            )
            print(
                f"Previous best loss: {frame_encoder_manager.best_loss * 1e3 :.6f}"
            )
            print(
                f"New best loss     : {loop_results.loss.cpu().item() * 1e3 :.6f}"
            )

            frame_encoder_manager.set_best_loss(loop_results.loss.cpu().item())

            # Save best results
            with open(f"{prefix_save}results_best.tsv", "w") as f_out:
                f_out.write(
                    loop_results.pretty_string(show_col_name=True, mode="all")
                    + "\n"
                )

            frame_encoder.save(
                path_best_frame_enc,
                frame_encoder_manager=frame_encoder_manager,
            )

            frame_encoder.set_to_eval()
            frame_encoder_output = frame_encoder.forward(
                reference_frames=[ref_i.data for ref_i in frame.refs_data],
                quantizer_noise_type="none",
                quantizer_type="hardround",
                AC_MAX_VAL=-1,
                flag_additional_outputs=True,
            )

            dec_path = f"{prefix_save}decoded-{frame.seq_name}"
            dec_img = frame_encoder_output.decoded_image
            if frame.data.frame_data_type == "rgb":
                if frame.data.bitdepth == 8:
                    write_png(dec_img, dec_path + ".png")
                else:
                    write_ppm(dec_img, dec_path + ".ppm", norm=True)
            # YUV
            else:
                write_yuv(
                    dec_img,
                    frame.data.bitdepth,
                    frame.data.frame_data_type,
                    dec_path + ".yuv",
                    norm=True,
                )
        # We have not beaten our record... But we still want to save the
        # updated frame_encoder_manager indicating that we've done one
        # additional loop
        else:
            # Load the best frame encoder
            best_frame_encoder, _ = load_frame_encoder(path_best_frame_enc)

            # Re-save it with the frame_encoder_manager
            best_frame_encoder.save(
                path_best_frame_enc,
                frame_encoder_manager=frame_encoder_manager,
            )

        print("End of training loop\n\n")

        if _is_job_over(
            start_time=start_time, max_duration_job_min=job_duration_min
        ):
            return TrainingExitCode.REQUEUE

    return TrainingExitCode.END


def get_ref_data(
    frame: Frame,
    coding_structure: CodingStructure,
) -> List[FrameData]:
    """Return a list of the (decoded) reference frames. The decoded data
    are obtained by recursively inferring the already learned FrameEncoder.

    Args:
        frame: The frame whose reference(s) we want.

    Returns:
        The decoded reference frames.
    """
    all_ref_data = []

    for ref_display_idx in frame.index_references:
        ref_frame = coding_structure.get_frame_from_display_order(
            ref_display_idx
        )
        assert ref_frame is not None

        # TODO: code duplication here
        # For now, reference are necessary a .yuv file
        dec_ref_path = (
            f"{_get_frame_path_prefix(ref_display_idx)}"
            "decoded-"
            f"{frame.seq_name}"
            ".yuv"
        )

        assert os.path.isfile(dec_ref_path), (
            f"Cannot find the decoded reference {dec_ref_path}.\n"
            "Hint: make sure that you have already encoded the frame "
            f"{ref_frame.frame_type}{ref_frame.display_order} using "
            f"--coding_idx={ref_frame.coding_order}"
        )

        # ! idx_display_order = 0 because this is a single-frame video
        # ! which directly corresponds to the reference frame.
        ref_data = load_frame_data_from_file(dec_ref_path, idx_display_order=0)

        # Load the references from the already stored
        ref_frame.set_frame_data(ref_data)

        all_ref_data.append(ref_frame.data)

    return all_ref_data


def _get_frame_path_prefix(frame_display_order: int) -> str:
    """Return a string with a prefix that should be appended to every file
    linked to a frame.

    Args:
        frame_display_order: Display order of the frame

    Returns:
        The prefix
    """
    return str(frame_display_order).zfill(4) + "-"


def frame_to_device(frame: Frame, device: POSSIBLE_DEVICE) -> Frame:
    """
    Push a frame and its reference to a device

    Args:
        frame: Frame to be pushed to a device
        device: Required device

    Returns:
        Frame: Frame pushed to a device
    """

    if frame.data is not None:
        frame_data_to_device(frame.data, device)

    for index_ref in range(len(frame.refs_data)):
        if frame.refs_data[index_ref] is not None:
            frame_data_to_device(frame.refs_data[index_ref], device)

    return frame


def frame_data_to_device(
    frame_data: FrameData, device: POSSIBLE_DEVICE
) -> FrameData:
    """Push the data attribute to the relevant device **in place**.

    Args:
        frame_data: The data to be pushed
        device: The device on which the frame data should be pushed.

    Returns:
        FrameData: The data on the correct device

    """
    from enc.io.format.yuv import yuv_dict_to_device

    if frame_data.frame_data_type == "yuv420":
        frame_data.data = yuv_dict_to_device(frame_data.data, device)
    else:
        frame_data.data = frame_data.data.to(device)

    return frame_data
