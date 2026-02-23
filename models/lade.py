import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class LADE(nn.Module):
    """Linear Adaptive Instance Denormalization"""

    def __init__(self, channels, use_sn=False):
        super(LADE, self).__init__()
        self.conv = nn.Conv2d(channels, channels,
                              kernel_size=1, stride=1, padding=0, bias=True)
        if use_sn:
            self.conv = spectral_norm(self.conv)
        self.eps = 1e-5

    def forward(self, x):
        # x is [B, C, H, W]
        tx = self.conv(x)

        # Calculate statistics
        t_mean = tx.mean(dim=[2, 3], keepdim=True)
        t_var = tx.var(dim=[2, 3], unbiased=False, keepdim=True)
        t_sigma = torch.sqrt(t_var + self.eps)

        in_mean = x.mean(dim=[2, 3], keepdim=True)
        in_var = x.var(dim=[2, 3], unbiased=False, keepdim=True)
        in_sigma = torch.sqrt(in_var + self.eps)

        x_in = (x - in_mean) / in_sigma

        out = x_in * t_sigma + t_mean
        return out
