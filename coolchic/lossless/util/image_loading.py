from typing import assert_never

import cv2
import lossless.util.color_transform as color_transform
import numpy as np
import torch
from lossless.util.colorspace import RGB, PossibleColorspace, YCoCg
from lossless.util.device import PossibleDevice


def load_image_as_tensor(image_path, device: PossibleDevice, color_space: PossibleColorspace):
    im = cv2.imread(filename=image_path)
    assert im is not None, f"Failed to read image {image_path}"
    im = im[:, :, ::-1].copy()

    c_bitdepths = color_transform.ColorBitdepths
    match color_space:
        case YCoCg():
            im_ycocg = color_transform.rgb_to_ycocg(im)
            im_rgb = color_transform.ycocg_to_rgb(im_ycocg)
            assert (
                np.abs(im.astype(np.int32) - im_rgb).max() == 0
            ), "Color transform is not lossless"
            im_tensor = (torch.from_numpy(im_ycocg).float() / 255.0).permute((2, 0, 1))[
                None,
            ]  # Change to CxHxW
            c_bitdepths = color_transform.YCoCgBitdepths()
        case RGB():
            # print("Using RGB color space")
            im_tensor = (torch.from_numpy(im).float() / 255.0).permute((2, 0, 1))[
                None,
            ]  # Change to CxHxW
            c_bitdepths = color_transform.RGBBitdepths()
        case _:
            assert_never(color_space)
    
    return im_tensor.to(device.materialize()), c_bitdepths
