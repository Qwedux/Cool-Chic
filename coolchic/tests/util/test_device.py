from collections.abc import Sequence
from dataclasses import dataclass

import pytest
import torch
from lossless.util.device import (CpuDevice, CudaZeroDevice, PossibleDevice,
                                  get_device)


@dataclass(frozen=True, slots=True, kw_only=True)
class MaterializeADTcases:
    test_input: PossibleDevice
    expected_test_output: torch.device


@dataclass(frozen=True, slots=True, kw_only=True)
class GetADTcases:
    test_input: torch.device | str
    expected_test_output: PossibleDevice


_MATERIALIZE_CASES: Sequence[MaterializeADTcases] = (
    MaterializeADTcases(test_input=CpuDevice(), expected_test_output=torch.device("cpu")),
    MaterializeADTcases(test_input=CudaZeroDevice(), expected_test_output=torch.device("cuda:0")),
)

_GET_DEVICE_OK_CASES: Sequence[GetADTcases] = (
    GetADTcases(test_input="cpu", expected_test_output=CpuDevice()),
    GetADTcases(test_input=torch.device("cpu"), expected_test_output=CpuDevice()),
    GetADTcases(test_input="cuda:0", expected_test_output=CudaZeroDevice()),
    GetADTcases(test_input=torch.device("cuda:0"), expected_test_output=CudaZeroDevice()),
    GetADTcases(test_input=torch.device("cuda", 0), expected_test_output=CudaZeroDevice()),
)


def _materialize_adt_case_id(case: MaterializeADTcases) -> str:
    return f"{type(case.test_input).__name__}->{case.expected_test_output}"


def _get_adt_case_id(case: GetADTcases) -> str:
    case_input = case.test_input
    expected_case_output = repr(case_input) if isinstance(case_input, str) else str(case_input)
    return f"{expected_case_output}->{type(case.expected_test_output).__name__}"


@pytest.mark.parametrize("case", _MATERIALIZE_CASES, ids=_materialize_adt_case_id)
def test_materialize_device(case: MaterializeADTcases) -> None:
    assert case.test_input.materialize() == case.expected_test_output


@pytest.mark.parametrize("case", _GET_DEVICE_OK_CASES, ids=_get_adt_case_id)
def test_get_device_ok(case: GetADTcases) -> None:
    assert get_device(case.test_input) == case.expected_test_output


def test_get_device_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Invalid device type"):
        get_device(torch.device("meta"))
