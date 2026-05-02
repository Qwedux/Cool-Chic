from dataclasses import dataclass


@dataclass(frozen=True)
class KumaraswamyType:
    pass

@dataclass(frozen=True)
class GaussianType:
    pass

@dataclass(frozen=True)
class NoQuantizationNoiseType:
    pass

POSSIBLE_QUANTIZATION_NOISE_TYPE = KumaraswamyType | GaussianType | NoQuantizationNoiseType