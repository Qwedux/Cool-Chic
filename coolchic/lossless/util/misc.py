# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

import os
from datetime import datetime

import psutil
import torch


def mem_info(strinfo: str = "Memory allocated") -> None:
    """Convenient printing of the current CPU / GPU memory allocated,
    prefixed by strinfo.

    Args:
        strinfo (str, optional): Printing prefix. Defaults to "Memory allocated".
    """
    mem_cpu = psutil.Process().memory_info().rss
    mem_cpu_GB = mem_cpu / (1024.0 * 1024.0 * 1024.0)

    mem_gpu = torch.cuda.memory_allocated("cuda:0")
    mem_gpu_GB = mem_gpu / (1024.0 * 1024.0 * 1024.0)

    str_display = f"| {strinfo:30s} cpu:{mem_cpu_GB:7.3f} GB --- gpu:{mem_gpu_GB:7.3f} GB |"
    L = len(str_display)
    print(f'{" "*100}{"-"*L}')
    print(f'{" "*100}{str_display}')
    print(f'{" "*100}{"-"*L}')


def timestamp_string(time: float | None = None) -> str:
    if time is not None:
        return datetime.fromtimestamp(time).strftime("%Y_%m_%d__%H_%M_%S_")
    else:
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


def safe_get_from_nested_lists(a: list, indices: list[int], default=0):
    """Safely get a nested element from a arbitrarily nested list of lists.
    Add try-except outside if you have more indices than levels of nesting.
    We do not provide it here for faster execution.
    """
    x = a
    for idx in indices:
        if 0 <= idx < len(x):
            x = x[idx]
        else:
            return default
    return x
