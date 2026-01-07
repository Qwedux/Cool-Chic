import numpy as np

VALID_COLORSPACE = ["RGB", "YCoCg"]


class ColorBitdepths:
    def __init__(self) -> None:
        self.bitdepths = []
        self.scaling_factors = []
        self.bins = []
        self.ranges_int = []


class RGBBitdepths(ColorBitdepths):
    def __init__(self) -> None:
        self.bitdepths = [8, 8, 8]
        self.scaling_factors = [255, 255, 255]
        self.bins = [256, 256, 256]
        self.ranges_int = [[0, 255], [0, 255], [0, 255]]


class YCoCgBitdepths(ColorBitdepths):
    def __init__(self) -> None:
        self.bitdepths = [8, 9, 9]
        self.scaling_factors = [255, 255, 255]
        self.bins = [256, 512, 512]
        self.ranges_int = [[0, 255], [-256, 255], [-256, 255]]


class LatentBitdepths(ColorBitdepths):
    def __init__(self) -> None:
        self.bitdepths = [6]
        self.scaling_factors = [1]
        self.bins = [64]
        self.ranges_int = [[-31, 32]]


def rgb_to_ycocg(img):
    r, g, b = (
        img[..., 0].astype(np.int32),
        img[..., 1].astype(np.int32),
        img[..., 2].astype(np.int32),
    )
    co = r - b
    t = b + (co >> 1)
    cg = g - t
    y = t + (cg >> 1)
    return np.stack((y, co, cg), axis=-1).astype(np.int32)


def ycocg_to_rgb(img):
    y, co, cg = img[..., 0], img[..., 1], img[..., 2]
    t = y - (cg >> 1)
    g = cg + t
    b = t - (co >> 1)
    r = b + co
    return np.stack((r, g, b), axis=-1).astype(np.int32)
