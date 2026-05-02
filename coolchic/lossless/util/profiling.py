from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TrainingProfilerArtifacts:
    trace_path: str
    memory_timeline_path: str
    key_averages_path: str


def create_training_profiler() -> torch.profiler.profile:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    return torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )


def export_training_profiler(
    profiler: Any,
    output_dir: str,
    file_prefix: str,
) -> TrainingProfilerArtifacts:
    os.makedirs(output_dir, exist_ok=True)

    trace_path = os.path.join(output_dir, f"{file_prefix}_chrome_trace.json")
    memory_timeline_path = os.path.join(output_dir, f"{file_prefix}_memory_timeline.html")
    key_averages_path = os.path.join(output_dir, f"{file_prefix}_key_averages.txt")

    profiler.export_chrome_trace(trace_path)
    profiler.export_memory_timeline(memory_timeline_path)

    sort_by = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    with open(key_averages_path, "w") as f:
        f.write(
            profiler.key_averages(group_by_input_shape=True).table(
                sort_by=sort_by,
                row_limit=200,
            )
        )

    return TrainingProfilerArtifacts(
        trace_path=trace_path,
        memory_timeline_path=memory_timeline_path,
        key_averages_path=key_averages_path,
    )
