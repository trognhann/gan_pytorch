import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class LADE(nn.Module):
    """Linear Adaptive Instance Denormalization - Optimized Version"""

    def __init__(self, channels, use_sn=False):
        super(LADE, self).__init__()
        conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv = spectral_norm(conv) if use_sn else conv
        self.eps = 1e-5

    def forward(self, x):
        tx = self.conv(x)

        t_var, t_mean = torch.var_mean(tx, dim=[2, 3], keepdim=True, unbiased=False)
        in_var, in_mean = torch.var_mean(x, dim=[2, 3], keepdim=True, unbiased=False)

        t_sigma = torch.sqrt(t_var + self.eps)
        in_sigma = torch.sqrt(in_var + self.eps)

        out = (x - in_mean) * (t_sigma / in_sigma) + t_mean
        
        return out