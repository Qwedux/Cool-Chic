import numpy as np


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
