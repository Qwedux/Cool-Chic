import torch
import torchac
from lossless.util.distribution import compute_logistic_cdfs

def encode(x: torch.Tensor, mu: torch.Tensor, scale: torch.Tensor):
    x_maxv = (1 << 8) - 1
    print(f"raw x: ",x.min(), x.max(), x.mean())
    x = torch.round(x_maxv * x).to(torch.int16).float() / x_maxv
    x_reshape = torch.round(x * ((1 << 8) - 1)).to(torch.int16).cpu()

    print(f"raw x_reshape: ",x_reshape.min(), x_reshape.max(), x_reshape.mean())
    byte_strings = []
    for i in range(3):
        symbols = x_reshape[:, i : i + 1, ...]
        print(
            f"Channel {i}: symbols shape {symbols.shape}, min {symbols.min()}, max {symbols.max()}"
        )
        cur_cdfs = compute_logistic_cdfs(
            mu[:, i : i + 1, ...], scale[:, i : i + 1, ...], 8
        ).cpu()
        print(cur_cdfs.shape)
        byte_strings.append(
            torchac.encode_int16_normalized_cdf(cur_cdfs, symbols)
        )
    return byte_strings


def decode(byte_strings: list, mu: torch.Tensor, scale: torch.Tensor):
    assert len(byte_strings) == 3
    x_maxv = (1 << 8) - 1

    _, _, h, w = mu.size()
    x_rec = torch.zeros(1, 3, h, w)

    # Channel 0 (Red)
    cur_cdfs_r = compute_logistic_cdfs(
        mu[:, :1, ...], scale[:, :1, ...], 8
    ).cpu()
    symbols_r = torchac.decode_int16_normalized_cdf(cur_cdfs_r, byte_strings[0])
    print(
        f"Decoded R: shape {symbols_r.shape}, min {symbols_r.min()}, max {symbols_r.max()}"
    )
    x_r = symbols_r.reshape(1, 1, h, w).float() / x_maxv
    x_rec[:, 0:1, ...] = x_r  # FIX: was using 3*i (which is 0), should be 0:1

    # Channel 1 (Green)
    cur_cdfs_g = compute_logistic_cdfs(
        mu[:, 1:2, ...], scale[:, 1:2, ...], 8
    ).cpu()
    symbols_g = torchac.decode_int16_normalized_cdf(cur_cdfs_g, byte_strings[1])
    print(
        f"Decoded G: shape {symbols_g.shape}, min {symbols_g.min()}, max {symbols_g.max()}"
    )
    x_g = symbols_g.reshape(1, 1, h, w).float() / x_maxv
    x_rec[:, 1:2, ...] = x_g  # FIX: was using 3*i+1 (which is 1), should be 1:2

    # Channel 2 (Blue)
    cur_cdfs_b = compute_logistic_cdfs(
        mu[:, 2:3, ...], scale[:, 2:3, ...], 8
    ).cpu()
    symbols_b = torchac.decode_int16_normalized_cdf(cur_cdfs_b, byte_strings[2])
    print(
        f"Decoded B: shape {symbols_b.shape}, min {symbols_b.min()}, max {symbols_b.max()}"
    )
    x_b = symbols_b.reshape(1, 1, h, w).float() / x_maxv
    x_rec[:, 2:3, ...] = x_b  # FIX: was using 3*i+2 (which is 2), should be 2:3

    return x_rec