import copy

import torch
from lossless.component.coolchic import CoolChicEncoder
from lossless.component.core.quantizer import (
    POSSIBLE_QUANTIZATION_NOISE_TYPE, POSSIBLE_QUANTIZER_TYPE)
from lossless.configs.presets import MODULE_TO_OPTIMIZE, Preset, TrainerPhase
from lossless.nnquant.quantizemodel import quantize_model
from lossless.training.loss import LossFunctionOutput, loss_function
from lossless.training.manager import ImageEncoderManager
from lossless.training.test import test
from lossless.util.color_transform import ColorBitdepths
from lossless.util.logger import TrainingLogger
from torch.nn.utils import clip_grad_norm_


# Custom scheduling function for the soft rounding temperature and the noise parameter
def _linear_schedule(
    initial_value: float, final_value: float, cur_itr: float, max_itr: float
) -> float:
    """Linearly schedule a function to go from initial_value at cur_itr = 0 to
    final_value when cur_itr = max_itr.

    Args:
        initial_value (float): Initial value for the scheduling
        final_value (float): Final value for the scheduling
        cur_itr (float): Current iteration index
        max_itr (float): Total number of iterations

    Returns:
        float: The linearly scheduled value @ iteration number cur_itr
    """
    assert cur_itr >= 0 and cur_itr <= max_itr, (
        f"Linear scheduling from 0 to {max_itr} iterations"
        " except to have a current iterations between those two values."
        f" Found cur_itr = {cur_itr}."
    )

    return cur_itr * (final_value - initial_value) / max_itr + initial_value


def _train_single_phase(
    model: CoolChicEncoder,
    target_image: torch.Tensor,
    image_encoder_manager: ImageEncoderManager,
    training_phase: TrainerPhase,
    logger: TrainingLogger,
    encoder_logs_best: LossFunctionOutput,
) -> tuple[CoolChicEncoder, LossFunctionOutput]:
    """Train a ``CoolChicEncoder`` for a single training phase with parameters
    defined in ``training_phase``."""

    model.train()
    model.compile(fullgraph=True)
    # check if the model is compiled

    best_model = model.get_param()

    # ------ Build the list of parameters to optimize
    # Iteratively construct the list of required parameters.
    parameters_to_optimize = []
    for cur_module_to_optimize in training_phase.optimized_module:
        match cur_module_to_optimize:
            case "all":
                parameters_to_optimize += [*model.parameters()]
            case "arm":
                parameters_to_optimize += [*model.arm.parameters()]
            case "arm_image":
                parameters_to_optimize += [*model.image_arm.parameters()]
            case "upsampling":
                parameters_to_optimize += [*model.upsampling.parameters()]
            case "synthesis":
                parameters_to_optimize += [*model.synthesis.parameters()]
            case "latent":
                parameters_to_optimize += [*model.latent_grids.parameters()]
            case other:
                raise ValueError(
                    f"Unknown module to optimize: {other}. "
                    f"Available modules are: {MODULE_TO_OPTIMIZE}"
                )

    optimizer = torch.optim.Adam(parameters_to_optimize, lr=training_phase.lr)
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())

    if training_phase.schedule_lr:
        # TODO: I'd like to use an explicit function for this scheduler
        learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_phase.max_itr // training_phase.freq_valid,
            eta_min=0.00001,
            last_epoch=-1,
        )
    else:
        learning_rate_scheduler = None

    # Initialize soft rounding temperature and noise parameter
    cur_softround_temperature = _linear_schedule(
        training_phase.softround_temperature[0],
        training_phase.softround_temperature[1],
        0,
        training_phase.max_itr,
    )
    cur_softround_temperature = torch.tensor(cur_softround_temperature, device=model.device)

    cur_noise_parameter = _linear_schedule(
        training_phase.noise_parameter[0],
        training_phase.noise_parameter[1],
        0,
        training_phase.max_itr,
    )
    cur_noise_parameter = torch.tensor(cur_noise_parameter, device=model.device)

    cnt_record = 0
    # Slightly faster to create the list once outside of the loop
    all_parameters = [x for x in model.parameters()]

    for cnt in range(training_phase.max_itr):

        # ------- Patience mechanism
        if cnt - cnt_record > training_phase.patience:
            if training_phase.schedule_lr:
                # reload the best model so far
                logger.log_training(
                    "Reseting the model with the best model and smaller learning rate"
                )
                model.set_param(best_model)
                optimizer.load_state_dict(best_optimizer_state)
                assert learning_rate_scheduler is not None
                current_lr = learning_rate_scheduler.state_dict()["_last_lr"][0]
                # actualise the best optimizer lr with current lr
                for g in optimizer.param_groups:
                    g["lr"] = current_lr

                cnt_record = cnt
            else:
                # exceeding the patience level ends the phase
                break

        # ------- Actual optimization
        # This is slightly faster than optimizer.zero_grad()
        for param in all_parameters:
            param.grad = None

        # forward / backward
        out_forward = model(
            image=target_image,
            quantizer_noise_type=training_phase.quantizer_noise_type,
            quantizer_type=training_phase.quantizer_type,
            soft_round_temperature=cur_softround_temperature,
            noise_parameter=cur_noise_parameter,
        )

        loss_function_output = loss_function(
            out_forward,
            target_image,
            colorspace_bitdepths=image_encoder_manager.colorspace_bitdepths,
        )
        loss_function_output.loss.backward()

        clip_grad_norm_(all_parameters, 1e-1, norm_type=2.0, error_if_nonfinite=False)
        optimizer.step()

        image_encoder_manager.iterations_counter += 1

        # ------- Validation
        # Each freq_valid iteration or at the end of the phase, compute validation loss and log stuff
        if ((cnt + 1) % training_phase.freq_valid == 0) or (cnt + 1 == training_phase.max_itr):
            #  a. Update iterations counter and training time and test model

            # b. Test the model and check whether we've beaten our record
            encoder_logs = test(
                model,
                target_image,
                image_encoder_manager,
            )

            flag_new_record = False
            new_rate = encoder_logs.rate_img_bpd + encoder_logs.rate_latent_bpd
            best_old_rate = encoder_logs_best.rate_img_bpd + encoder_logs_best.rate_latent_bpd
            logger.log_training(f"new rate is: {new_rate}, old rate is: {best_old_rate}")
            if new_rate < best_old_rate:
                # A record must have at least -0.001 bpp or + 0.001 dB. A smaller improvement
                # does not matter.
                delta_psnr = encoder_logs_best.rate_img_bpd - encoder_logs.rate_img_bpd
                delta_loss = encoder_logs_best.loss - encoder_logs.loss
                flag_new_record = delta_psnr > 0.001 or delta_loss > 0.001
                logger.log_training(f"delta loss: {delta_loss}, flag new record: {flag_new_record}")

            if flag_new_record:
                logger.log_training("Found new best model!")
                # Save best model
                best_model = model.get_param()
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())

                # ========================= reporting ========================= #
                this_phase_psnr_gain = encoder_logs.rate_img_bpd - encoder_logs_best.rate_img_bpd

                log_new_record = ""
                log_new_record += f"{this_phase_psnr_gain:+6.3f} db"
                # ========================= reporting ========================= #

                # Update new record
                encoder_logs_best = encoder_logs
                cnt_record = cnt

            # # Show column name a single time
            # additional_data = {
            #     "lr": f"{start_lr if not cosine_scheduling_lr else learning_rate_scheduler.get_last_lr()[0]:.4f}",
            #     "optim": ",".join(optimized_module),
            #     "patience": (patience - cnt + cnt_record)
            #     // frequency_validation,
            #     "q_type": f"{quantizer_type:10s}",
            #     "sr_temp": f"{cur_softround_temperature:.3f}",
            #     "n_type": f"{quantizer_noise_type:12s}",
            #     "noise": f"{cur_noise_parameter:.2f}",
            #     "record": log_new_record,
            # }

            logger.log_training(f"Iteration: {cnt+1}, " + str(encoder_logs))

            # Update soft rounding temperature and noise_parameter
            cur_softround_temperature = _linear_schedule(
                training_phase.softround_temperature[0],
                training_phase.softround_temperature[1],
                cnt,
                training_phase.max_itr,
            )
            cur_softround_temperature = torch.tensor(cur_softround_temperature, device=model.device)

            cur_noise_parameter = _linear_schedule(
                training_phase.noise_parameter[0],
                training_phase.noise_parameter[1],
                cnt,
                training_phase.max_itr,
            )
            cur_noise_parameter = torch.tensor(cur_noise_parameter, device=model.device)

            if training_phase.schedule_lr:
                assert learning_rate_scheduler is not None
                learning_rate_scheduler.step()

            model.train()

    # At the end of the training, we load the best model
    model.set_param(best_model)
    if training_phase.quantize_model:
        model = quantize_model(
            model,
            target_image,
            image_encoder_manager,
            logger,
        )

    return model, encoder_logs_best
