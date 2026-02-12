import torch
import torch.nn as nn
from net.common import ConvBlock, ResBlock, ExternalAttention, Upsample


class Generator(nn.Module):
    def __init__(self, img_ch=3):
        super(Generator, self).__init__()

        # 1. Encoder
        self.head = ConvBlock(img_ch, 32, kernel_size=7, stride=1, padding=3)

        # Downsample
        self.down1 = ConvBlock(32, 64, stride=2)
        self.down2 = ConvBlock(64, 128, stride=2)

        # 2. Bottleneck with External Attention
        # Use a mix of ResBlocks and External Attention for lightweight design
        self.res1 = ResBlock(128)
        self.attn1 = ExternalAttention(128, S=32)
        self.res2 = ResBlock(128)
        self.attn2 = ExternalAttention(128, S=32)
        self.res3 = ResBlock(128)

        # 3. Decoder
        self.up1 = Upsample(128, 64)
        self.up_conv1 = ConvBlock(64, 64)

        self.up2 = Upsample(64, 32)
        self.up_conv2 = ConvBlock(32, 32)

        # 4. Double Tail

        # Support Tail (Coarse anime image)
        self.support_tail = nn.Sequential(
            ConvBlock(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

        # Main Tail (Refined anime image)
        self.main_tail = nn.Sequential(
            ConvBlock(32, 32, kernel_size=3, padding=1),
            # Refined details
            nn.Conv2d(32, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x, training=True):
        # Encoder
        x = self.head(x)
        x = self.down1(x)
        x = self.down2(x)

        # Bottleneck
        x = self.res1(x)
        x = self.attn1(x)
        x = self.res2(x)
        x = self.attn2(x)
        x = self.res3(x)

        # Decoder
        x = self.up1(x)
        x = self.up_conv1(x)
        x = self.up2(x)
        x = self.up_conv2(x)

        # Double Tail Outputs
        main_out = self.main_tail(x)

        if training:
            support_out = self.support_tail(x)
            return main_out, support_out

        return main_out
