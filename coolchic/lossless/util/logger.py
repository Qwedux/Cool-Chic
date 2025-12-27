import os
import time

import torch
from lossless.configs.config import start_print
from lossless.util.misc import timestamp_string


class TrainingLogger:
    def __init__(self, log_folder_path: str, image_name: str, debug_mode: bool = False, experiment_name: str = "kodak"):
        self.log_folder_path = log_folder_path
        self.debug_mode = debug_mode
        if self.debug_mode:
            self.log_folder_path += "debug/"
        else:
            self.log_folder_path += "full_runs/"
        self.log_folder_path += experiment_name + "/"
        self.train_logs_path = self.log_folder_path + "train_logs/"
        self.results_logs_path = self.log_folder_path + "results/"
        self.trained_models_path = self.log_folder_path + "trained_models/"

        self.image_name = image_name
        os.makedirs(self.log_folder_path, exist_ok=True)
        os.makedirs(self.train_logs_path, exist_ok=True)
        os.makedirs(self.results_logs_path, exist_ok=True)
        os.makedirs(self.trained_models_path, exist_ok=True)
        self.start_time = time.time()

        train_log_file_name = f"{timestamp_string(self.start_time)}_coolchic_{self.image_name}.log"
        results_log_file_name = f"{timestamp_string(self.start_time)}_coolchic_{self.image_name}.log"
        self.train_log_file_path = os.path.join(
            self.train_logs_path, train_log_file_name
        )
        self.results_log_file_path = os.path.join(
            self.results_logs_path, results_log_file_name
        )
        with open(self.results_log_file_path, "w") as f:
            f.write(start_print)
            f.write(
                f"Log file created at {timestamp_string(self.start_time)}\n"
            )
            f.write(f"Image name: {self.image_name}\n")
            f.write("\n")

    def log_result(self, message: str):
        print(message)
        with open(self.results_log_file_path, "a") as f:
            f.write(message + "\n")
        return

    def log_training(self, message: str):
        print(message)
        with open(self.train_log_file_path, "a") as f:
            f.write(message + "\n")
        return

    def save_model(self, model, total_bpd: float):
        # Save the model to disk only if not in debug mode (don't bother with 6.0 Loss models)
        if self.debug_mode:
            return
        model_file_name = f"{timestamp_string()}_trained_coolchic_{self.image_name}_img_rate_{total_bpd}.pth"
        model_file_path = os.path.join(
            self.trained_models_path, model_file_name
        )
        torch.save(model.state_dict(), model_file_path)
        self.log_result(
            f"Model saved at {model_file_path} with rate {total_bpd} bpd."
        )
        return
