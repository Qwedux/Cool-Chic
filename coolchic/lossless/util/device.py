from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeAlias, assert_never

import torch


@dataclass(frozen=True, slots=True)
class AbstractDevice(ABC):
    @abstractmethod
    @staticmethod
    def materialize() -> torch.device:
        raise NotImplementedError

@dataclass(frozen=True, slots=True)
class CudaZeroDevice(AbstractDevice):
    @staticmethod
    def materialize() -> torch.device:
        return torch.device("cuda:0")

@dataclass(frozen=True, slots=True)
class CpuDevice(AbstractDevice):
    @staticmethod
    def materialize() -> torch.device:
        return torch.device("cpu")

PossibleDevice: TypeAlias = CpuDevice | CudaZeroDevice

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
