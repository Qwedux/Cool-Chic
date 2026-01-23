import cv2
import lossless.util.color_transform as color_transform
import numpy as np
import torch


def load_image_as_tensor(image_path, device="cuda:0", color_space="YCoCg"):
    assert color_space in color_transform.VALID_COLORSPACE, f"Invalid color space {color_space}"
    im = cv2.imread(filename=image_path)
    assert im is not None, f"Failed to read image {image_path}"
    im = im[:, :, ::-1].copy()  # BGR to RGB

    c_bitdepths = color_transform.ColorBitdepths
    if color_space == "YCoCg":
        print("Using YCoCg color space")
        im_ycocg = color_transform.rgb_to_ycocg(im)
        im_rgb = color_transform.ycocg_to_rgb(im_ycocg)
        assert (
            np.abs(im.astype(np.int32) - im_rgb).max() == 0
        ), "Color transform is not lossless"
        im_tensor = (torch.from_numpy(im_ycocg).float() / 255.0).permute((2, 0, 1))[
            None,
        ]  # Change to CxHxW
        c_bitdepths = color_transform.YCoCgBitdepths()
    elif color_space == "RGB":
        print("Using RGB color space")
        im_tensor = (torch.from_numpy(im).float() / 255.0).permute((2, 0, 1))[
            None,
        ]  # Change to CxHxW
        c_bitdepths = color_transform.RGBBitdepths()
    else:
        raise ValueError(f"Invalid color space {color_space}")
    return im_tensor.to(device), c_bitdepths
