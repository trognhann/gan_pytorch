import torch.nn as nn
from .lade import LADE

class ConvLadeLrelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvLadeLrelu, self).__init__()
        if (kernel_size - stride) % 2 == 0:
            pad = (kernel_size - stride) // 2
            pad_left, pad_right, pad_top, pad_bottom = pad, pad, pad, pad
        else:
            pad = (kernel_size - stride) // 2
            pad_bottom, pad_right = pad, pad
            pad_top, pad_left = kernel_size - stride - \
                pad_bottom, kernel_size - stride - pad_right

        self.pad = nn.ReflectionPad2d(
            (pad_left, pad_right, pad_top, pad_bottom))
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=0, bias=True)
        self.lade = LADE(out_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.lade(x)
        return self.lrelu(x)