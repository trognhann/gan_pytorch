import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from .lade import LADE



def conv_block(in_channels, out_channels, kernel_size=3, stride=1, sn=True):
    if (kernel_size - stride) % 2 == 0:
        pad = (kernel_size - stride) // 2
        pad_left, pad_right, pad_top, pad_bottom = pad, pad, pad, pad
    else:
        pad = (kernel_size - stride) // 2
        pad_bottom, pad_right = pad, pad
        pad_top, pad_left = kernel_size - stride - \
            pad_bottom, kernel_size - stride - pad_right

    layers = [
        nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom)),
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=0, bias=False)
    ]
    if sn:
        layers[-1] = spectral_norm(layers[-1])
    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, ch=32, sn=True):
        super(Discriminator, self).__init__()

        self.conv_0 = conv_block(
            in_channels, ch, kernel_size=7, stride=1, sn=sn)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        channels = ch
        self.blocks = nn.ModuleList()
        for i in range(3):
            self.blocks.append(nn.Sequential(
                conv_block(channels, channels, kernel_size=3, stride=2, sn=sn),
                LADE(channels, use_sn=sn),
                nn.LeakyReLU(0.2, inplace=True),

                conv_block(channels, channels * 2,
                           kernel_size=3, stride=1, sn=sn),
                LADE(channels * 2, use_sn=sn),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            channels = channels * 2

        self.final_conv = conv_block(
            channels, 1, kernel_size=1, stride=1, sn=sn)

    def forward(self, x):
        x = self.lrelu(self.conv_0(x))
        for block in self.blocks:
            x = block(x)
        return self.final_conv(x)
