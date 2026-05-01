from dataclasses import dataclass
from typing import TypeAlias, assert_never

import torch


@dataclass(frozen=True, slots=True)
class CudaZeroDevice:
    pass

@dataclass(frozen=True, slots=True)
class CpuDevice:
    pass

PossibleDevice: TypeAlias = CpuDevice | CudaZeroDevice

def materialize_device(device: PossibleDevice) -> torch.device:
    match device:
        case CudaZeroDevice():
            return torch.device("cuda:0")
        case CpuDevice():
            return torch.device("cpu")
        case _:
            assert_never(device)

def get_device(device: torch.device | str) -> PossibleDevice:
    match device:
        case str():
            return get_device(torch.device(device))
        case torch.device():
            match device.type:
                case "cpu":
                    return CpuDevice()
                case "cuda":
                    return CudaZeroDevice()
                case _:
                    raise ValueError(f"Invalid device type: {device.type}")
        case _:
            assert_never(device)
