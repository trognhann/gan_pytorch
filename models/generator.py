import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import ExternalAttention_v3
from .conv_block import ConvLadeLrelu


class Generator(nn.Module):
    def __init__(self, in_channels=3):
        super(Generator, self).__init__()

        # Base encoder
        self.base_x0 = ConvLadeLrelu(
            in_channels, 32, kernel_size=7, stride=1)  # 256
        self.base_x1_1 = ConvLadeLrelu(32, 32, kernel_size=3, stride=2)  # 128
        self.base_x1_2 = ConvLadeLrelu(32, 64, kernel_size=3, stride=1)

        self.base_x2_1 = ConvLadeLrelu(64, 64, kernel_size=3, stride=2)  # 64
        self.base_x2_2 = ConvLadeLrelu(64, 128, kernel_size=3, stride=1)

        self.base_x3_1 = ConvLadeLrelu(128, 128, kernel_size=3, stride=2)  # 32
        self.base_x3_2 = ConvLadeLrelu(128, 128, kernel_size=3, stride=1)

        # Support Tail
        self.s_attn = ExternalAttention_v3(128)
        self.s_x4_1 = ConvLadeLrelu(128, 128, kernel_size=3, stride=1)
        self.s_x4_2 = ConvLadeLrelu(128, 128, kernel_size=3, stride=1)

        self.s_x5_1 = ConvLadeLrelu(128, 64, kernel_size=3, stride=1)
        self.s_x5_2 = ConvLadeLrelu(64, 64, kernel_size=3, stride=1)

        self.s_x6_1 = ConvLadeLrelu(64, 32, kernel_size=3, stride=1)
        self.s_x6_2 = ConvLadeLrelu(32, 32, kernel_size=3, stride=1)

        self.s_final_pad = nn.ReflectionPad2d(3)
        self.s_final = nn.Conv2d(32, 3, kernel_size=7, stride=1, padding=0)

        # Main Tail
        self.m_attn = ExternalAttention_v3(128)
        self.m_x4_1 = ConvLadeLrelu(128, 128, kernel_size=3, stride=1)
        self.m_x4_2 = ConvLadeLrelu(128, 128, kernel_size=3, stride=1)

        self.m_x5_1 = ConvLadeLrelu(128, 64, kernel_size=3, stride=1)
        self.m_x5_2 = ConvLadeLrelu(64, 64, kernel_size=3, stride=1)

        self.m_x6_1 = ConvLadeLrelu(64, 32, kernel_size=3, stride=1)
        self.m_x6_2 = ConvLadeLrelu(32, 32, kernel_size=3, stride=1)

        self.m_final_pad = nn.ReflectionPad2d(3)
        self.m_final = nn.Conv2d(32, 3, kernel_size=7, stride=1, padding=0)

    def _upsample(self, x):
        return F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)

    def forward(self, x, inference=False):
        # Base
        x0 = self.base_x0(x)
        x1 = self.base_x1_2(self.base_x1_1(x0))
        x2 = self.base_x2_2(self.base_x2_1(x1))
        x3 = self.base_x3_2(self.base_x3_1(x2))

        if not inference:
            # Support Tail
            s_x3 = self.s_attn(x3)
            s_x4 = self._upsample(s_x3)
            s_x4 = self.s_x4_1(s_x4)
            s_x4 = self.s_x4_2(s_x4 + x2)

            s_x5 = self._upsample(s_x4)
            s_x5 = self.s_x5_1(s_x5)
            s_x5 = self.s_x5_2(s_x5 + x1)

            s_x6 = self._upsample(s_x5)
            s_x6 = self.s_x6_1(s_x6)
            s_x6 = self.s_x6_2(s_x6 + x0)

            s_final = self.s_final(self.s_final_pad(s_x6))
            fake_s = torch.tanh(s_final)
        else:
            fake_s = None

        # Main Tail
        m_x3 = self.m_attn(x3)
        m_x4 = self._upsample(m_x3)
        m_x4 = self.m_x4_1(m_x4)
        m_x4 = self.m_x4_2(m_x4 + x2)

        m_x5 = self._upsample(m_x4)
        m_x5 = self.m_x5_1(m_x5)
        m_x5 = self.m_x5_2(m_x5 + x1)

        m_x6 = self._upsample(m_x5)
        m_x6 = self.m_x6_1(m_x6)
        m_x6 = self.m_x6_2(m_x6 + x0)

        m_final = self.m_final(self.m_final_pad(m_x6))
        fake_m = torch.tanh(m_final)

        if inference:
            return fake_m
        return fake_s, fake_m
