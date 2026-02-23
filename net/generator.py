import torch
import torch.nn as nn
import torch.nn.functional as F
from net.common import ConvBlock, ExternalAttention

class Generator(nn.Module):
    def __init__(self, img_ch=3):
        super(Generator, self).__init__()

        # ==========================================
        # 1. BASE ENCODER (Khớp 100% TF)
        # ==========================================
        # Input: (B, 3, 256, 256) -> Output: (B, 32, 256, 256)
        self.x0_conv = ConvBlock(img_ch, 32, kernel_size=7, stride=1, padding=3)

        # Output: (B, 64, 128, 128)
        self.x1_conv1 = ConvBlock(32, 32, kernel_size=3, stride=2, padding=1)
        self.x1_conv2 = ConvBlock(32, 64, kernel_size=3, stride=1, padding=1)

        # Output: (B, 128, 64, 64)
        self.x2_conv1 = ConvBlock(64, 64, kernel_size=3, stride=2, padding=1)
        self.x2_conv2 = ConvBlock(64, 128, kernel_size=3, stride=1, padding=1)

        # Output: (B, 128, 32, 32) - Điểm thắt (Bottleneck)
        self.x3_conv1 = ConvBlock(128, 128, kernel_size=3, stride=2, padding=1)
        self.x3_conv2 = ConvBlock(128, 128, kernel_size=3, stride=1, padding=1)


        # ==========================================
        # 2. SUPPORT TAIL (Khớp 100% TF)
        # ==========================================
        self.s_attn = ExternalAttention(128, S=64)

        # 32x32 -> 64x64
        self.s_up1_conv1 = ConvBlock(128, 128, kernel_size=3, stride=1, padding=1)
        self.s_up1_conv2 = ConvBlock(128, 128, kernel_size=3, stride=1, padding=1)

        # 64x64 -> 128x128
        self.s_up2_conv1 = ConvBlock(128, 64, kernel_size=3, stride=1, padding=1)
        self.s_up2_conv2 = ConvBlock(64, 64, kernel_size=3, stride=1, padding=1)

        # 128x128 -> 256x256
        self.s_up3_conv1 = ConvBlock(64, 32, kernel_size=3, stride=1, padding=1)
        self.s_up3_conv2 = ConvBlock(32, 32, kernel_size=3, stride=1, padding=1)

        self.s_out = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )


        # ==========================================
        # 3. MAIN TAIL (Khớp 100% TF)
        # ==========================================
        self.m_attn = ExternalAttention(128, S=64)

        # 32x32 -> 64x64
        self.m_up1_conv1 = ConvBlock(128, 128, kernel_size=3, stride=1, padding=1)
        self.m_up1_conv2 = ConvBlock(128, 128, kernel_size=3, stride=1, padding=1)

        # 64x64 -> 128x128
        self.m_up2_conv1 = ConvBlock(128, 64, kernel_size=3, stride=1, padding=1)
        self.m_up2_conv2 = ConvBlock(64, 64, kernel_size=3, stride=1, padding=1)

        # 128x128 -> 256x256
        self.m_up3_conv1 = ConvBlock(64, 32, kernel_size=3, stride=1, padding=1)
        self.m_up3_conv2 = ConvBlock(32, 32, kernel_size=3, stride=1, padding=1)

        self.m_out = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def upsample_bilinear(self, x):
        # Hàm resize tương đương tf.image.resize_images
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, training=True):
        # --- Base Encoder ---
        x0 = self.x0_conv(x)

        x1 = self.x1_conv1(x0)
        x1 = self.x1_conv2(x1)

        x2 = self.x2_conv1(x1)
        x2 = self.x2_conv2(x2)

        x3 = self.x3_conv1(x2)
        x3 = self.x3_conv2(x3)

        # --- Support Tail ---
        s_x3 = self.s_attn(x3)

        s_x4 = self.upsample_bilinear(s_x3)
        s_x4 = self.s_up1_conv1(s_x4)
        s_x4 = self.s_up1_conv2(s_x4 + x2)  # Skip Connection x2

        s_x5 = self.upsample_bilinear(s_x4)
        s_x5 = self.s_up2_conv1(s_x5)
        s_x5 = self.s_up2_conv2(s_x5 + x1)  # Skip Connection x1

        s_x6 = self.upsample_bilinear(s_x5)
        s_x6 = self.s_up3_conv1(s_x6)
        s_x6 = self.s_up3_conv2(s_x6 + x0)  # Skip Connection x0

        fake_s = self.s_out(s_x6)

        # --- Main Tail ---
        m_x3 = self.m_attn(x3)

        m_x4 = self.upsample_bilinear(m_x3)
        m_x4 = self.m_up1_conv1(m_x4)
        m_x4 = self.m_up1_conv2(m_x4 + x2)  # Skip Connection x2

        m_x5 = self.upsample_bilinear(m_x4)
        m_x5 = self.m_up2_conv1(m_x5)
        m_x5 = self.m_up2_conv2(m_x5 + x1)  # Skip Connection x1

        m_x6 = self.upsample_bilinear(m_x5)
        m_x6 = self.m_up3_conv1(m_x6)
        m_x6 = self.m_up3_conv2(m_x6 + x0)  # Skip Connection x0

        fake_m = self.m_out(m_x6)

        if training:
            return fake_m, fake_s

        return fake_m