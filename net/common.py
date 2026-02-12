import torch
import torch.nn as nn
import torch.nn.functional as F


class LADE(nn.Module):
    """
    Linearly Adaptive Denormalization (LADE)
    Modulates normalized features using spatially adaptive parameters derived from the input itself.
    """

    def __init__(self, num_features, eps=1e-5):
        super(LADE, self).__init__()
        self.eps = eps
        # Learnable parameters for modulation, spatially adaptive via 1x1 conv
        self.gamma_conv = nn.Conv2d(
            num_features, num_features, kernel_size=1, bias=True)
        self.beta_conv = nn.Conv2d(
            num_features, num_features, kernel_size=1, bias=True)

    def forward(self, x):
        # 1. Instance Normalization (but manual to allow modulation)
        # Calculate mean and var per instance/channel
        N, C, H, W = x.size()
        loss_x = x.view(N, C, -1)
        mean = loss_x.mean(dim=2).view(N, C, 1, 1)
        var = loss_x.var(dim=2, unbiased=False).view(N, C, 1, 1)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # 2. Generate spatially adaptive parameters from the input (self-modulation)
        # Note: In some variants, this might come from a separate 'style' encoding,
        # but for AnimeGAN which is image-to-image without external style code,
        # LADE often uses the feature map itself to preserve spatial info.
        gamma = self.gamma_conv(x)
        beta = self.beta_conv(x)

        # 3. Denormalization
        out = x_norm * (1 + gamma) + beta
        return out


class ExternalAttention(nn.Module):
    """
    External Attention Mechanism
    Uses learnable external memory units (M_k, M_v) instead of self-attention input keys/values.
    """

    def __init__(self, in_channels, out_channels=None, S=64):
        super(ExternalAttention, self).__init__()
        out_channels = out_channels or in_channels
        self.k_dim = S
        self.v_dim = S or in_channels  # Use S or keep dim

        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Linear(in_channels, S, bias=False)  # Memory Key
        self.v = nn.Linear(S, in_channels, bias=False)  # Memory Value
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        n, c, h, w = x.size()

        # Query
        q = self.conv1(x).view(n, c, h*w).permute(0, 2, 1)  # N, HW, C

        # External Attention (Q * K_ext^T)
        # k is (C, S) -> k.T is (S, C)??? No, nn.Linear(in, out) has weights (out, in).
        # We want (N, HW, C) @ (C, S) -> (N, HW, S)
        # nn.Linear applies x A^T + b.
        # If we use it as a learnable matrix M_k:
        # We process Q to get attention map.

        # Paper implementation typically:
        # attn = Q @ K_transposed
        # Q: (N, HW, C)
        # K (Memory): is a parameter of shape (S, C)
        # But here let's use nn.Linear to represent the projection against external memory.

        attn = self.k(q)  # (N, HW, S)
        attn = F.softmax(attn, dim=1)

        # Normalization (optional but common in External Attn)
        attn = attn / (1e-9 + attn.sum(dim=2, keepdim=True))

        # Weighted sum: Attn @ V
        # V (Memory): (S, C)
        out = self.v(attn)  # (N, HW, C)

        out = out.permute(0, 2, 1).view(n, c, h, w)
        out = self.conv2(out)
        return out + x  # Residual connection


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_lade=True, activation=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, bias=False)
        self.norm = LADE(
            out_channels) if use_lade else nn.InstanceNorm2d(out_channels)
        self.act = nn.LeakyReLU(
            0.2, inplace=True) if activation else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, channels, use_lade=True):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels, use_lade=use_lade)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.norm = LADE(channels) if use_lade else nn.InstanceNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.norm(out)
        return x + out


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.out_channels = out_channels

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2,
                            mode='bilinear', align_corners=True)
        out = self.conv(out)
        return out
