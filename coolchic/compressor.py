from logging import config
from torch import nn
import torch.nn.functional as F
import torch
from torch import Tensor
from PIL import Image
import numpy as np
import constriction
import struct
import torch.nn.parallel as parallel


from .architectures import ModelConfig


def _laplace_cdf(x: Tensor, expectation: Tensor, scale: Tensor) -> Tensor:
    """Compute the laplace cumulative evaluated in x. All parameters
    must have the same dimension.
    Re-implemented here coz it is faster than calling the Laplace distribution
    from torch.distributions.

    Args:
        x (Tensor): Where the cumulative if evaluated.
        expectation (Tensor): Expectation.
        scale (Tensor): Scale

    Returns:
        Tensor: CDF(x, mu, scale)
    """
    shifted_x = x - expectation
    return 0.5 - 0.5 * (shifted_x).sign() * torch.expm1(
        -(shifted_x).abs() / scale
    )


def calculate_laplace_probability_distribution(mu, b):
    """Calculate Laplace probability distribution for arithmetic coding.
    Works for any shape of mu and b, adds one dimension at the end for the probability axis.
    """
    # Create the base tensor of quantized values
    new_tensor = torch.linspace(0, 255, steps=256, device=mu.device)  # [256]
    new_shape = (*mu.shape, 256)  # add one dimension at the end
    new_tensor = new_tensor.view(*([1] * mu.ndim), 256).expand(*new_shape)

    # Compute boundaries for each bin
    x_minus = new_tensor - 0.5
    x_plus = new_tensor + 0.5

    # Expand mu and b with one trailing dimension
    mu_expanded = mu.unsqueeze(-1)
    b_expanded = b.unsqueeze(-1)

    # Laplace CDF difference between bin edges (use the laplace cdf function)
    cdf_minus = _laplace_cdf(x_minus, mu_expanded, b_expanded)
    cdf_plus = _laplace_cdf(x_plus, mu_expanded, b_expanded)

    prob_t = cdf_plus - cdf_minus
    prob_t = torch.clamp_min(prob_t, 2 ** (-16))
    prob_t = prob_t / prob_t.sum(dim=-1, keepdim=True)

    return prob_t


def compute_loss(x, laplace_params):
    """Compute the compression loss for a batch of images."""
    mu, scale = laplace_params[:, :, :, 0], laplace_params[:, :, :, 1]

    x_flat = x.flatten(2, 3).permute(0, 2, 1)
    cdf_plus = _laplace_cdf(x_flat + 0.5, mu, scale)
    cdf_minus = _laplace_cdf(x_flat - 0.5, mu, scale)

    proba = torch.clamp_min(
        cdf_plus - cdf_minus,
        min=2**-16,  # No value can cost more than 16 bits.
    )

    rate = -torch.log2(proba)  # bits per pixel
    loss = rate.sum()

    return loss


class Compressor(nn.Module):
    def __init__(
        self,
        ups_model_config: ModelConfig,
        arm_model_config: ModelConfig,
        n_scales=5,
        n_thresholds: int = 10,
        context_window: int = 7,
        channels: int = 3,
        device: str = "cpu",
    ):
        super(Compressor, self).__init__()
        self.n_scales = n_scales
        self.n_thresholds = n_thresholds
        self.context_window = context_window
        self.channels = channels
        self.device = device

        self.arm_list = nn.ModuleList(
            [
                ARM(
                    arm_model_config,
                    context_window=context_window,
                    channels=channels,
                    n_thresholds=n_thresholds,
                    device=device,
                )
                for _ in range(n_scales)
            ]
            + [
                ARM(
                    arm_model_config,
                    context_window=context_window,
                    channels=channels,
                    n_thresholds=n_thresholds,
                    cond_ups=False,
                    device=device,
                )
            ]
        )
        self.ups_list = nn.ModuleList(
            [
                UPS(
                    ups_model_config,
                    context_window=context_window,
                    channels=channels,
                    n_thresholds=n_thresholds,
                    device=device,
                )
                for _ in range(n_scales)
            ]
        )

        # self.arm_list = nn.ModuleList([ARM(context_window=context_window, channels=channels, n_thresholds=n_thresholds) for _ in range(n_scales)]+[ARM(context_window=context_window, channels=channels, n_thresholds=n_thresholds, cond_ups=False)])
        # self.ups_list = nn.ModuleList([UPS(context_window=context_window, channels=channels, n_thresholds=n_thresholds) for _ in range(n_scales)])
        # self.mixing_parameter = nn.Parameter(torch.zeros(2))

        self.thresholds = self.spaced_values(n_thresholds)

    def forward(self, x):
        assert x.shape[2] >= 2 ** (self.n_scales - 1) and x.shape[3] >= 2 ** (
            self.n_scales
        ), "x must have at least 2**(n_scales-1) height and width"
        B, C, H, W = x.shape
        x_multiscale, shapes = self.multiscale(x)

        laplace_params = self.arm_list[-1](self.prepare_x(x_multiscale[-1]))
        loss = dict()
        loss["arm_0"] = compute_loss(x_multiscale[-1], laplace_params) / (
            B * H * W * C
        )

        for i in range(self.n_scales):
            x_ups = self.ups_list[-i - 1](self.prepare_x(x_multiscale[-i - 1]))
            x_ups = x_ups[:, :, : shapes[-i - 2][2], : shapes[-i - 2][3]]
            laplace_params = self.arm_list[-i - 2](
                self.prepare_x(x_multiscale[-i - 2]),
                self.prepare_x(x_ups.detach()),
            )
            loss["arm_" + str(i + 1)] = compute_loss(
                x_multiscale[-i - 2], laplace_params
            ) / (B * H * W * C)
            loss["ups_" + str(i + 1)] = (
                ((x_ups - x_multiscale[-i - 2]) / 255.0) ** 2
            ).mean()

        mu = laplace_params[:, :, :, 0]
        scale = laplace_params[:, :, :, 1]
        return loss, mu, scale

    def forward_arm_loss(self, x, level):
        """
        Calculate ARM loss at a specific level.

        Args:
            x: Input tensor
            level: Level index (0 to n_scales for ARM)

        Returns:
            loss: The computed ARM loss for the specified level
            mu: Mean parameters from the ARM model
            scale: Scale parameters from the ARM model
        """
        assert (
            level <= self.n_scales
        ), "Level must be less than or equal to n_scales"
        assert level >= 0, "Level must be greater than or equal to 0"
        assert x.shape[2] >= 2 ** (self.n_scales - 1) and x.shape[3] >= 2 ** (
            self.n_scales
        ), "x must have at least 2**(n_scales-1) height and width"
        B, C, H, W = x.shape
        x_multiscale, shapes = self.multiscale(x)

        if level == 0:
            # ARM at the finest level (level n_scales)
            laplace_params = self.arm_list[-1](self.prepare_x(x_multiscale[-1]))
            loss = compute_loss(x_multiscale[-1], laplace_params)
        else:
            # ARM at intermediate levels (requires UPS input)
            # First get the UPS output for this level (in eval mode)
            ups_model = self.ups_list[-level]
            ups_model.eval()
            with torch.no_grad():
                x_ups = ups_model(self.prepare_x(x_multiscale[-level]))
                x_ups = x_ups[
                    :, :, : shapes[-level - 1][2], : shapes[-level - 1][3]
                ]

            # Then compute ARM loss
            laplace_params = self.arm_list[-level - 1](
                self.prepare_x(x_multiscale[-level - 1]),
                self.prepare_x(x_ups.detach()),
            )
            loss = compute_loss(x_multiscale[-level - 1], laplace_params)

        mu = laplace_params[:, :, :, 0]
        scale = laplace_params[:, :, :, 1]
        return loss / (B * H * W * C), mu, scale

    def forward_ups_loss(self, x, level):
        """
        Calculate UPS loss at a specific level.

        Args:
            x: Input tensor
            level: Level index (1 to n_scales for UPS, no UPS on level 0 or n_scales)

        Returns:
            loss: The computed UPS loss for the specified level
        """
        assert (
            level <= self.n_scales
        ), "Level must be less than or equal to n_scales"
        assert level >= 1, "Level must be greater than or equal to 1"
        assert x.shape[2] >= 2 ** (self.n_scales - 1) and x.shape[3] >= 2 ** (
            self.n_scales
        ), "x must have at least 2**(n_scales-1) height and width"

        B, C, H, W = x.shape
        x_multiscale, shapes = self.multiscale(x)

        # UPS at level (level-1) in the original indexing
        x_ups = self.ups_list[-level](self.prepare_x(x_multiscale[-level]))
        x_ups = x_ups[:, :, : shapes[-level - 1][2], : shapes[-level - 1][3]]

        # UPS loss is MSE between predicted and target
        target = x_multiscale[-level - 1]
        # print(x_ups[0, 0, 0, 0].item(), target[0, 0, 0, 0].item())
        ups_loss = (((x_ups - target) / 255.0) ** 2).mean()
        return ups_loss

    def compute_bicubic_ups_loss(self, x, level):
        """
        Calculate Bicubic UPS loss at a specific level.

        Args:
            x: Input tensor
            level: Level index (1 to n_scales for UPS, no UPS on level 0 or n_scales)
        """
        assert (
            level <= self.n_scales
        ), "Level must be less than or equal to n_scales"
        assert level >= 1, "Level must be greater than or equal to 1"
        assert x.shape[2] >= 2 ** (self.n_scales - 1) and x.shape[3] >= 2 ** (
            self.n_scales
        ), "x must have at least 2**(n_scales-1) height and width"

        B, C, H, W = x.shape
        x_multiscale, shapes = self.multiscale(x)

        # Bicubic UPS at level (level-1) in the original indexing
        x_ups = F.interpolate(
            x_multiscale[-level],
            scale_factor=2,
            mode="bicubic",
            align_corners=False,
        )
        x_ups = x_ups[:, :, : shapes[-level - 1][2], : shapes[-level - 1][3]]

        # Bicubic UPS loss is MSE between predicted and target
        target = x_multiscale[-level - 1]
        bicubic_ups_loss = (((x_ups - target) / 255.0) ** 2).mean()
        return bicubic_ups_loss

    def multiscale(self, x):
        shapes = [x.shape]
        x_multiscale = [x]
        for i in range(self.n_scales):
            x_i = x_multiscale[-1]

            h, w = x_i.shape[2], x_i.shape[3]
            pad_h = 1 if x_i.shape[2] % 2 == 1 else 0
            pad_w = 1 if w % 2 == 1 else 0
            if pad_h or pad_w:
                x_i = F.pad(x_i, (0, pad_w, 0, pad_h), mode="reflect")

            x_scale = F.avg_pool2d(x_i, kernel_size=2)
            x_scale = x_scale.int().float()
            x_multiscale.append(x_scale)
            shapes.append(x_multiscale[-1].shape)
        return x_multiscale, shapes

    def prepare_x(self, x):
        x = torch.stack(
            [
                torch.where(
                    x > float(i), torch.ones_like(x), torch.zeros_like(x)
                )
                for i in self.thresholds
            ],
            dim=2,
        )
        x = x.flatten(1, 2)
        return x

    def spaced_values(self, n, low=0, high=255):
        # Divide range into n+1 intervals and take the midpoints
        edges = np.linspace(low, high, n + 1)
        mids = (edges[:-1] + edges[1:]) / 2
        return mids

    def encode_per_pixel(self, data_path, output_path):
        x = Image.open(data_path)
        if x.mode == "RGB":
            x = (
                torch.tensor(list(x.getdata()))
                .view(x.height, x.width, 3)
                .permute(2, 0, 1)
            )
        else:
            x = (
                torch.tensor(list(x.getdata()))
                .view(x.height, x.width)
                .unsqueeze(0)
            )
        x = x.unsqueeze(0).float()

        B, C, H, W = x.shape
        x_multiscale, shapes = self.multiscale(x)
        enc = constriction.stream.stack.AnsCoder()
        bits_theoretical = 0

        with torch.no_grad():
            for i in range(self.n_scales):
                x_ups = self.ups_list[i](self.prepare_x(x_multiscale[i + 1]))
                x_ups = x_ups[:, :, : shapes[i][2], : shapes[i][3]]
                # print(x_multiscale[i+1], x_ups[0, 0, 0, 0, 0].item())

                scale_theoretical_bits = 0
                for wh in range(shapes[i][2] * shapes[i][3]):
                    # Calculate mixing for this specific pixel
                    h = wh // shapes[i][3]
                    w = wh % shapes[i][3]

                    laplace_params = self.arm_list[i].forward_one_pixel(
                        h,
                        w,
                        self.prepare_x(x_multiscale[i]),
                        self.prepare_x(x_ups[:, :, :, :, 0]),
                    )  # B, H*W, self.channels, 2

                    mu_new = laplace_params[:, :, :, 0] * mixing_parameter[
                        0
                    ] + x_ups[0, :, h, w, 0].unsqueeze(0).unsqueeze(1) * (
                        1 - mixing_parameter[0]
                    )
                    scale_new = laplace_params[:, :, :, 1] * mixing_parameter[
                        1
                    ] + x_ups[0, :, h, w, 1].unsqueeze(0).unsqueeze(1) * (
                        1 - mixing_parameter[1]
                    )
                    laplace_params_pixel = (
                        torch.stack([mu_new, scale_new], dim=1)
                        .unsqueeze(0)
                        .unsqueeze(1)
                    )  # Shape: (1, 1, C, 2)
                    prob = calculate_laplace_probability_distribution(
                        laplace_params_pixel[:, :, :, 0],
                        laplace_params_pixel[:, :, :, 1],
                    )

                    for c in range(self.channels):
                        sym = (
                            (
                                x_multiscale[i]
                                .flatten(2, 3)
                                .permute(0, 2, 1)[0, -wh - 1, -c - 1]
                            )
                            .int()
                            .item()
                        )
                        prob_t = prob[0, 0, c].flatten()  # Ensure it's 1D
                        scale_theoretical_bits += -torch.log2(
                            prob_t[sym]
                        ).item()
                        model = constriction.stream.model.Categorical(
                            prob_t.detach().cpu().numpy(), perfect=False
                        )
                        enc.encode_reverse(sym, model)
                        # print(sym, prob_t[sym].item(), laplace_params_pixel[0, 0, c, 0].item(), laplace_params_pixel[0, 0, c, 1].item(), x_ups[0, :, h, w, 0].item())
                bits_theoretical += scale_theoretical_bits

            scale_theoretical_bits = 0
            for wh in range(shapes[-1][2] * shapes[-1][3]):
                h = wh // shapes[i][3]
                w = wh % shapes[i][3]
                laplace_params = self.arm_list[-1].forward_one_pixel(
                    h, w, self.prepare_x(x_multiscale[-1])
                )
                prob = calculate_laplace_probability_distribution(
                    laplace_params[:, :, :, 0], laplace_params[:, :, :, 1]
                )
                for c in range(self.channels):
                    sym = (
                        (
                            x_multiscale[-1]
                            .flatten(2, 3)
                            .permute(0, 2, 1)[0, -wh - 1, -c - 1]
                        )
                        .int()
                        .item()
                    )
                    # prob_t = prob[0, -wh-1, -c-1]
                    prob_t = prob[0, 0, c].flatten()  # Ensure it's 1D
                    scale_theoretical_bits += -torch.log2(prob_t[sym]).item()
                    model = constriction.stream.model.Categorical(
                        prob_t.detach().cpu().numpy(), perfect=False
                    )
                    enc.encode_reverse(sym, model)
            bits_theoretical += scale_theoretical_bits

        bitstream = enc.get_compressed()
        bitstream.tofile(output_path)
        with open(output_path, "rb") as f:
            original_data = f.read()
        with open(output_path, "wb") as f:
            # Pack two 32-bit integers into binary
            f.write(struct.pack("iii", H, W, C))
            f.write(original_data)

        print(
            f"Theoretical bits per sub pixel: {bits_theoretical/float(W*H*C)}"
        )

    def encode(self, data_path, output_path):
        x = Image.open(data_path)
        if x.mode == "RGB":
            x = (
                torch.tensor(list(x.getdata()))
                .view(x.height, x.width, 3)
                .permute(2, 0, 1)
            )
        else:
            x = (
                torch.tensor(list(x.getdata()))
                .view(x.height, x.width)
                .unsqueeze(0)
            )
        x = x.unsqueeze(0).float()

        B, C, H, W = x.shape
        x_multiscale, shapes = self.multiscale(x)
        enc = constriction.stream.stack.AnsCoder()
        bits_theoretical = 0

        with torch.no_grad():
            for i in range(self.n_scales):
                x_ups = self.ups_list[i](self.prepare_x(x_multiscale[i + 1]))
                x_ups = x_ups[:, :, : shapes[i][2], : shapes[i][3]]
                # print(x_multiscale[i+1], x_ups[0, 0, 0, 0, 0].item())
                laplace_params = self.arm_list[i](
                    self.prepare_x(x_multiscale[i]),
                    self.prepare_x(x_ups[:, :, :, :]),
                )  # B, H*W, self.channels

                scale_theoretical_bits = 0
                for wh in range(shapes[i][2] * shapes[i][3]):
                    h = wh // shapes[i][3]
                    w = wh % shapes[i][3]

                    # Use ARM parameters directly (no mixing)
                    laplace_params_pixel = (
                        laplace_params[0, -wh - 1, :, :]
                        .unsqueeze(0)
                        .unsqueeze(1)
                    )  # Shape: (1, 1, C, 2)
                    prob = calculate_laplace_probability_distribution(
                        laplace_params_pixel[:, :, :, 0],
                        laplace_params_pixel[:, :, :, 1],
                    )

                    for c in range(self.channels):
                        sym = (
                            (
                                x_multiscale[i]
                                .flatten(2, 3)
                                .permute(0, 2, 1)[0, -wh - 1, -c - 1]
                            )
                            .int()
                            .item()
                        )
                        prob_t = prob[0, 0, c].flatten()  # Ensure it's 1D
                        scale_theoretical_bits += -torch.log2(
                            prob_t[sym]
                        ).item()
                        model = constriction.stream.model.Categorical(
                            prob_t.detach().cpu().numpy(), perfect=False
                        )
                        enc.encode_reverse(sym, model)
                        # print(sym, prob_t[sym].item(), laplace_params_pixel[0, 0, c, 0].item(), laplace_params_pixel[0, 0, c, 1].item(), x_ups[0, :, h, w, 0].item())
                bits_theoretical += scale_theoretical_bits
            laplace_params = self.arm_list[-1](self.prepare_x(x_multiscale[-1]))

            prob = calculate_laplace_probability_distribution(
                laplace_params[:, :, :, 0], laplace_params[:, :, :, 1]
            )
            scale_theoretical_bits = 0
            for wh in range(shapes[-1][2] * shapes[-1][3]):
                for c in range(self.channels):
                    sym = (
                        (
                            x_multiscale[-1]
                            .flatten(2, 3)
                            .permute(0, 2, 1)[0, -wh - 1, -c - 1]
                        )
                        .int()
                        .item()
                    )
                    prob_t = prob[0, -wh - 1, -c - 1]
                    scale_theoretical_bits += -torch.log2(prob_t[sym]).item()
                    model = constriction.stream.model.Categorical(
                        prob_t.detach().cpu().numpy(), perfect=False
                    )
                    enc.encode_reverse(sym, model)
            bits_theoretical += scale_theoretical_bits

        bitstream = enc.get_compressed()
        bitstream.tofile(output_path)
        with open(output_path, "rb") as f:
            original_data = f.read()
        with open(output_path, "wb") as f:
            # Pack two 32-bit integers into binary
            f.write(struct.pack("iii", H, W, C))
            f.write(original_data)

        print(
            f"Theoretical bits per sub pixel: {bits_theoretical/float(W*H*C)}"
        )

    def decode(self, bitstream_path, output_path):
        with open(bitstream_path, "rb") as f:
            header = f.read(12)  # 3 integers * 4 bytes each
            H, W, C = struct.unpack("iii", header)
        bitstream = np.fromfile(bitstream_path, dtype=np.uint32, offset=12)
        dec = constriction.stream.stack.AnsCoder(bitstream)

        x = -torch.ones(1, C, H, W)
        x_multiscale, shapes = self.multiscale(x)
        with torch.no_grad():
            for h in range(shapes[-1][2]):
                for w in range(shapes[-1][3]):
                    # print(x_multiscale[-1])
                    laplace_params = self.arm_list[-1].forward_one_pixel(
                        h, w, self.prepare_x(x_multiscale[-1])
                    )
                    prob = calculate_laplace_probability_distribution(
                        laplace_params[:, :, :, 0], laplace_params[:, :, :, 1]
                    )
                    for c in range(C):
                        prob_array = (
                            prob[0, 0, c].detach().cpu().flatten().numpy()
                        )
                        model = constriction.stream.model.Categorical(
                            prob_array, perfect=False
                        )
                        decoded_char = torch.tensor(
                            dec.decode(model, 1)[0]
                        ).float()
                        # print(decoded_char, prob_array[int(decoded_char.item())].item())
                        x_multiscale[-1][0, c, h, w] = decoded_char
            for i in range(self.n_scales):
                x_ups = self.ups_list[-i - 1](
                    self.prepare_x(x_multiscale[-i - 1])
                )
                x_ups = x_ups[:, :, : shapes[-i - 2][2], : shapes[-i - 2][3]]
                # print(x_multiscale[-i-2], x_ups[0, 0, 0, 0, 0].item())
                for h in range(shapes[-i - 2][2]):
                    for w in range(shapes[-i - 2][3]):
                        laplace_params = self.arm_list[
                            -i - 2
                        ].forward_one_pixel(
                            h,
                            w,
                            self.prepare_x(x_multiscale[-i - 2]),
                            self.prepare_x(x_ups[:, :, :, :]),
                        )
                        # Use ARM parameters directly (no mixing)
                        prob = calculate_laplace_probability_distribution(
                            laplace_params[:, :, :, 0],
                            laplace_params[:, :, :, 1],
                        )
                        for c in range(C):
                            prob_array = (
                                prob[0, 0, c].detach().cpu().flatten().numpy()
                            )
                            model = constriction.stream.model.Categorical(
                                prob_array, perfect=False
                            )
                            decoded_char = torch.tensor(
                                dec.decode(model, 1)[0]
                            ).float()
                            x_multiscale[-i - 2][0, c, h, w] = decoded_char
                            # print(decoded_char, prob_array[int(decoded_char.item())].item(), laplace_params[0, 0, c, 0].item(), laplace_params[0, 0, c, 1].item(), x_ups[0, :, h, w, 0].item())
        x = x_multiscale[0]
        x = x.cpu().numpy()
        # print(x.min(), x.max(), x.shape)
        x = x.astype(np.uint8)

        if x.shape[0] == 1:  # Grayscale
            x = x[
                0, 0
            ]  # Remove batch and channel dimensions: (1, 1, H, W) -> (H, W)
            x = Image.fromarray(x, mode="L")
        else:  # RGB
            x = x[0].transpose(
                1, 2, 0
            )  # Remove batch dim and convert from CHW to HWC: (1, C, H, W) -> (H, W, C)
            x = Image.fromarray(x, mode="RGB")
        x.save(output_path)


class UPS(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        context_window: int = 3,
        channels: int = 3,
        n_thresholds: int = 10,
        device: str = "cpu",
    ):
        super(UPS, self).__init__()
        self.context_window = context_window
        self.channels = channels
        self.n_thresholds = n_thresholds
        self.model_config = model_config

        input_size = channels * n_thresholds * context_window**2

        self.model = model_config.create(
            input_size=input_size, num_classes=channels * 4, device=device
        )

        # self.devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        # if len(self.devices) > 1:
        #     self.replicas = torch.nn.parallel.replicate(self.model, self.devices)
        # else:
        #     self.replicas = [self.model]

    # def forward(self, x):
    #     B, C, H, W = x.shape
    #     x = self.unroll_x(x) # (B*H*W, input_size)
    #     x = self.model(x)
    #     x = x.reshape(B, H, W, self.channels, 2, 2, 2) # (B, H, W, C, 2, 2, 2)
    #     x = x.permute(0, 3, 1, 4, 2, 5, 6) # (B, C, H, 2, W, 2, 2)
    #     x = x.flatten(2, 3)
    #     x = x.flatten(3, 4) # (B, C, H*2, W*2, 2)
    #     x = torch.stack([
    #         x[:, :, :, :, 0] * 255.0,
    #         torch.clamp(x[:, :, :, :, 1] / (1.0 - x[:, :, :, :, 1] + 1e-1) * 100.0, max=1000.0, min=1e-8)
    #     ], dim=4)
    #     return x

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.unroll_x(x)  # (B*H*W, input_size)
        x = self.model(x)
        x = x.reshape(B, H, W, self.channels, 2, 2)  # (B, H, W, C, 2, 2)
        x = x.permute(0, 3, 1, 4, 2, 5)  # (B, C, H, 2, W, 2)
        x = x.flatten(2, 3)
        x = x.flatten(3, 4)  # (B, C, H*2, W*2)
        x = x * 255.0
        return x

    # def forward(self, x):
    #     B, C, H, W = x.shape
    #     devices = self.devices

    #     x = self.unroll_x(x)
    #     x_chunks = torch.chunk(x, len(devices))

    #     # Each input must be a tuple
    #     inputs = [(chunk.to(devices[i]),) for i, chunk in enumerate(x_chunks)]

    #     # Recreate replicas each forward if training
    #     replicas = (
    #         parallel.replicate(self.model, devices)
    #         if self.training else getattr(self, "replicas", None) or parallel.replicate(self.model, devices)
    #     )
    #     if not self.training:
    #         self.replicas = replicas  # cache them for inference

    #     # Run in parallel
    #     outputs = parallel.parallel_apply(replicas, inputs, devices=devices)

    #     # Gather results back to first GPU
    #     x = torch.cat([o.to(devices[0]) for o in outputs], dim=0)

    #     # --- continue with your logic ---
    #     x = x.reshape(B, H, W, self.channels, 2, 2, 2)
    #     x = x.permute(0, 3, 1, 4, 2, 5, 6)
    #     x = x.flatten(2, 3)
    #     x = x.flatten(3, 4)
    #     x = torch.stack([
    #         x[:, :, :, :, 0] * 255.0,
    #         torch.clamp(
    #             x[:, :, :, :, 1] / (1.0 - x[:, :, :, :, 1] + 1e-1) * 100.0,
    #             max=1000.0, min=1e-8
    #         )
    #     ], dim=4)
    #     return x

    def unroll_x(self, x):
        B, C, H, W = x.shape
        pad_size = self.context_window // 2
        x_padded = F.pad(x, (pad_size, pad_size, pad_size, pad_size))
        patches = F.unfold(
            x_padded, kernel_size=self.context_window
        )  # (B, C*K, H*W)
        patches = patches.transpose(1, 2)  # (B, H*W, C*K)
        unrolled_x = patches.flatten(0, 1)  # (B*H*W, input_size)

        return unrolled_x


class ARM(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        context_window: int = 7,
        cond_ups: bool = True,
        channels: int = 3,
        n_thresholds: int = 10,
        device: str = "cpu",
    ):
        super(ARM, self).__init__()
        assert context_window % 2 == 1, "Context window must be odd"
        self.context_window = context_window
        self.cond_ups = cond_ups
        self.channels = channels
        self.n_thresholds = n_thresholds
        self.model_config = model_config

        # Each channel contributes the same number of causal features
        if self.cond_ups:
            input_size = channels * n_thresholds * self.context_window**2
        else:
            input_size = (
                channels * n_thresholds * ((self.context_window**2 - 1) // 2)
            )

        self.model = model_config.create(
            input_size=input_size, num_classes=channels * 2, device=device
        )

        # self.devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        # if len(self.devices) > 1:
        #     self.replicas = torch.nn.parallel.replicate(self.model, self.devices)
        # else:
        #     self.replicas = [self.model]

    def forward(self, x, x_ups=None):
        B, C, H, W = x.shape
        x = self.causal_unroll(x)  # (B*H*W, -1)
        if self.cond_ups:
            x_ups = self.ups_unroll(x_ups)  # (B*H*W, -1)
            x_ups = x_ups[:, x.shape[1] :]
            x = torch.cat([x, x_ups], dim=1)

        x = self.model(x)  # (B*H*W, channels*2)
        x = x.reshape(B, H * W, self.channels, 2)

        # scale outputs
        x = torch.stack(
            [
                x[:, :, :, 0] * 255.0,
                torch.clamp(
                    x[:, :, :, 1] / (1.0 - x[:, :, :, 1] + 1e-1) * 100.0,
                    max=1000.0,
                    min=1e-8,
                ),
            ],
            dim=3,
        )
        return x

    # def forward_one_pixel(self, h, w, x, x_ups=None):
    #     B, C, H, W = x.shape
    #     x = self.forward(x, x_ups)
    #     x = x[:, h*W+w]
    #     x = x.unsqueeze(1)
    #     return x

    def forward_one_pixel(self, h, w, x, x_ups=None):
        B, C, H, W = x.shape
        x = self.causal_unroll_one_pixel(x, h, w)
        if self.cond_ups:
            x_ups = self.ups_unroll_one_pixel(x_ups, h, w)
            x_ups = x_ups[:, x.shape[1] :]
            x = torch.cat([x, x_ups], dim=1)
        x = self.model(x)
        x = x.reshape(B, 1, self.channels, 2)

        x = torch.stack(
            [
                x[:, :, :, 0] * 255.0,
                torch.clamp(
                    x[:, :, :, 1] / (1.0 - x[:, :, :, 1] + 1e-1) * 100.0,
                    max=1000.0,
                    min=1e-8,
                ),
            ],
            dim=3,
        )
        return x

    def causal_unroll(self, x):
        """
        x: (B, C, H, W)
        Returns:
            unrolled_x: (B, H*W, C * kept_elements)
        """
        B, C, H, W = x.shape
        pad_size = self.context_window // 2
        x_padded = F.pad(x, (pad_size, pad_size, pad_size, pad_size))
        patches = F.unfold(
            x_padded, kernel_size=self.context_window
        )  # (B, C*K, H*W)
        patches = patches.transpose(1, 2)  # (B, H*W, C*K)

        K = self.context_window * self.context_window
        # indices of causal positions in the flattened window
        center_idx = K // 2
        causal_mask = torch.arange(K, device=x.device) < center_idx  # (K,)

        # apply mask separately for each channel
        causal_mask = causal_mask.repeat(C)  # (C*K,)
        unrolled_x = patches[:, :, causal_mask]  # (B, H*W, C*((K-1)//2))

        unrolled_x = unrolled_x.flatten(0, 1)  # (B*H*W, input_size)

        return unrolled_x

    def causal_unroll_one_pixel(self, x, h, w):
        B, C, H, W = x.shape
        unrolled_x = self.causal_unroll(x)
        unrolled_x = unrolled_x[h * W + w].unsqueeze(0)
        return unrolled_x

    def ups_unroll(self, x):
        B, C, H, W = x.shape
        pad_size = self.context_window // 2
        x_padded = F.pad(x, (pad_size, pad_size, pad_size, pad_size))
        patches = F.unfold(
            x_padded, kernel_size=self.context_window
        )  # (B, C*K, H*W)
        patches = patches.transpose(1, 2)  # (B, H*W, C*K)
        unrolled_x = patches.flatten(0, 1)  # (B*H*W, input_size)

        return unrolled_x

    def ups_unroll_one_pixel(self, x, h, w):
        B, C, H, W = x.shape
        pad_size = self.context_window // 2
        x_padded = F.pad(x, (pad_size, pad_size, pad_size, pad_size))
        patch = x_padded[
            :, :, h : h + 2 * pad_size + 1, w : w + 2 * pad_size + 1
        ]  # (B, C, context_window, context_window)
        patch = patch.flatten(2, 3)  # (B, C, context_window*context_window)
        patch = patch.flatten(1, 2)  # (B, C*context_window*context_window)
        return patch
