import torch
import torch.nn as nn
import torch.nn.functional as F


class CogVideoXDownsample3D(nn.Module):
    # Todo: Wait for paper relase.
    r"""
    A 3D Downsampling layer using in [CogVideoX]() by Tsinghua University & ZhipuAI

    Args:
        in_channels (`int`):
            Number of channels in the input image.
        out_channels (`int`):
            Number of channels produced by the convolution.
        kernel_size (`int`, defaults to `3`):
            Size of the convolving kernel.
        stride (`int`, defaults to `2`):
            Stride of the convolution.
        padding (`int`, defaults to `0`):
            Padding added to all four sides of the input.
        compress_time (`bool`, defaults to `False`):
            Whether or not to compress the time dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 0,
        compress_time: bool = False,
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.compress_time = compress_time

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.compress_time:
            batch_size, channels, frames, height, width = x.shape

            # (batch_size, channels, frames, height, width) -> (batch_size, height, width, channels, frames) -> (batch_size * height * width, channels, frames)
            x = x.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, channels, frames)

            if x.shape[-1] % 2 == 1:
                x_first, x_rest = x[..., 0], x[..., 1:]
                if x_rest.shape[-1] > 0:
                    # (batch_size * height * width, channels, frames - 1) -> (batch_size * height * width, channels, (frames - 1) // 2)
                    x_rest = F.avg_pool1d(x_rest, kernel_size=2, stride=2)

                x = torch.cat([x_first[..., None], x_rest], dim=-1)
                # (batch_size * height * width, channels, (frames // 2) + 1) -> (batch_size, height, width, channels, (frames // 2) + 1) -> (batch_size, channels, (frames // 2) + 1, height, width)
                x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(0, 3, 4, 1, 2)
            else:
                # (batch_size * height * width, channels, frames) -> (batch_size * height * width, channels, frames // 2)
                x = F.avg_pool1d(x, kernel_size=2, stride=2)
                # (batch_size * height * width, channels, frames // 2) -> (batch_size, height, width, channels, frames // 2) -> (batch_size, channels, frames // 2, height, width)
                x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(0, 3, 4, 1, 2)

        # Pad the tensor
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        batch_size, channels, frames, height, width = x.shape
        # (batch_size, channels, frames, height, width) -> (batch_size, frames, channels, height, width) -> (batch_size * frames, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
        x = self.conv(x)
        # (batch_size * frames, channels, height, width) -> (batch_size, frames, channels, height, width) -> (batch_size, channels, frames, height, width)
        x = x.reshape(batch_size, frames, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
        return x
