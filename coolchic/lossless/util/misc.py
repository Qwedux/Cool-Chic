import os
from datetime import datetime

def timestamp_string() -> str:
    return datetime.now().strftime("%Y_%m_%d__%H_%M_%S_")

def clean_workdir(workdir_path: str) -> None:
    """
    Clean the working directory by removing all files and subdirectories.

    Args:
        workdir_path (str): The path to the working directory to clean.
    Returns:
        None
    """
    if os.path.exists(workdir_path):
        print(f"Removing {workdir_path}...")
        for file in os.listdir(workdir_path):
            file_path = os.path.join(workdir_path, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

