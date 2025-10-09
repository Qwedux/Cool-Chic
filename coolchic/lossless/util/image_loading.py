import numpy as np
import torch
import cv2
from lossless.util.color_transform import rgb_to_ycocg, ycocg_to_rgb


def load_image_as_tensor(image_path, device="cuda:0"):
    im = cv2.imread(filename=image_path)
    assert im is not None, f"Failed to read image {image_path}"
    im = im[:, :, ::-1].copy()  # BGR to RGB
    im_ycocg = rgb_to_ycocg(im)
    im_rgb = ycocg_to_rgb(im_ycocg)
    assert (
        np.abs(im.astype(np.int32) - im_rgb).max() == 0
    ), "Color transform is not lossless"
    im_tensor = (torch.from_numpy(im_ycocg).float() / 255.0).permute((2, 0, 1))[
        None,
    ]  # Change to CxHxW
    return im_tensor.to(device)
