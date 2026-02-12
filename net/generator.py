import torch
import torch.nn as nn
from net.common import ConvBlock, ResBlock, ExternalAttention, Upsample


class Generator(nn.Module):
    def __init__(self, img_ch=3):
        super(Generator, self).__init__()

        # 1. Encoder (Shared)
        # x0: (B, 32, H, W)
        self.enc1 = ConvBlock(img_ch, 32, kernel_size=7, stride=1, padding=3)

        # x1: (B, 64, H/2, W/2)
        self.down1 = ConvBlock(32, 64, stride=2)

        # x2: (B, 128, H/4, W/4)
        self.down2 = ConvBlock(64, 128, stride=2)

        # x3: (B, 256, H/8, W/8) - Bottleneck Input
        self.down3 = ConvBlock(128, 256, stride=2)

        # 2. Support Tail
        self.s_attn = ExternalAttention(256, S=64)

        # Upsample 256 -> 128. Combine with x2
        self.s_upsample1 = Upsample(256, 128)
        self.s_up1 = ConvBlock(128, 128)

        # Upsample 128 -> 64. Combine with x1
        self.s_upsample2 = Upsample(128, 64)
        self.s_up2 = ConvBlock(64, 64)

        # Upsample 64 -> 32. Combine with x0
        self.s_upsample3 = Upsample(64, 32)
        self.s_up3 = ConvBlock(32, 32)

        self.s_out = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

        # 3. Main Tail
        self.m_attn = ExternalAttention(256, S=64)

        # Upsample 256 -> 128. Combine with x2
        self.m_upsample1 = Upsample(256, 128)
        self.m_up1 = ConvBlock(128, 128)

        # Upsample 128 -> 64. Combine with x1
        self.m_upsample2 = Upsample(128, 64)
        self.m_up2 = ConvBlock(64, 64)

        # Upsample 64 -> 32. Combine with x0
        self.m_upsample3 = Upsample(64, 32)
        self.m_up3 = ConvBlock(32, 32)

        self.m_out = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x, training=True):
        # Encoder
        x0 = self.enc1(x)       # 32
        x1 = self.down1(x0)     # 64
        x2 = self.down2(x1)     # 128
        x3 = self.down3(x2)     # 256

        # Support Tail
        s = self.s_attn(x3)
        s = self.s_upsample1(s)  # 256 -> 128
        s = self.s_up1(s + x2)  # Skip x2

        s = self.s_upsample2(s)  # 128 -> 64
        s = self.s_up2(s + x1)  # Skip x1

        s = self.s_upsample3(s)  # 64 -> 32
        s = self.s_up3(s + x0)  # Skip x0

        fake_s = self.s_out(s)

        # Main Tail
        m = self.m_attn(x3)
        m = self.m_upsample1(m)  # 256 -> 128
        m = self.m_up1(m + x2)  # Skip x2

        m = self.m_upsample2(m)  # 128 -> 64
        m = self.m_up2(m + x1)  # Skip x1

        m = self.m_upsample3(m)  # 64 -> 32
        m = self.m_up3(m + x0)  # Skip x0

        fake_m = self.m_out(m)

        if training:
            return fake_m, fake_s

        return fake_m
