from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NewType, Sequence

from config.preset import SubmoduleConfig

# ==============================================================================
# Encoding General
# ==============================================================================
ImageIndex = NewType("ImageIndex", int)
ExperimentName = NewType("ExperimentName", str)
LogPath = NewType("LogPath", Path)
DatasetPath = NewType("DatasetPath", Path)

# ==============================================================================
# Training
# ==============================================================================
@dataclass(frozen=True)
class UseAsInitWeights:
    pass

@dataclass(frozen=True)
class EvaluateOnly:
    pass

PretrainedModelUse = UseAsInitWeights | EvaluateOnly
PretrainedModelPath = NewType("PretrainedModelPath", Path)

# ==============================================================================
# Architecture
# ==============================================================================
LatentEncoderGain = NewType("LatentEncoderGain", int)
LatentFeatureDimensionSize = NewType("LatentFeatureDimensionSize", int)
LatentFeaturesPerResolution = Sequence[LatentFeatureDimensionSize]
UpsamplingKernelSize = NewType("UpsamplingKernelSize", int)
UpsamplingPreconcatKernelSize = NewType("UpsamplingPreconcatKernelSize", int)
UseColorRegression = NewType("UseColorRegression", bool)
ComputationalGraphConfig = Sequence[SubmoduleConfig]

# ==============================================================================
# Debugging
# ==============================================================================
PrintDetailedArchitecture = NewType("PrintDetailedArchitecture", bool)
Profile = NewType("Profile", bool)
ProfilingOutputDir = NewType("ProfilingOutputDir", Path)