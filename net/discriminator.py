import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm


class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=32):
        super(Discriminator, self).__init__()

        # 1. Initial Layer: Kernel 7, Stride 1 (Matches TF)
        self.head = nn.Sequential(
            spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=7,
                          stride=1, padding=3, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 2. Downsampling Loop (Matches TF: 3 blocks)
        self.body = nn.Sequential()
        channel = ndf
        for i in range(3):
            # Block i: Stride 2 -> LADE/Norm -> Act -> Stride 1 -> LADE/Norm -> Act
            self.body.add_module(
                f"block_{i}", self._make_discriminator_block(channel, channel * 2))
            channel = channel * 2

        # 3. Final Logit: Kernel 1x1 (Matches TF)
        self.tail = spectral_norm(
            nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0, bias=False))

    def _make_discriminator_block(self, in_ch, out_ch):
        """
        Structure: 
        Conv(s2) -> LADE -> Act -> Conv(s1) -> LADE -> Act
        """
        return nn.Sequential(
            # Conv Stride 2
            spectral_norm(nn.Conv2d(in_ch, in_ch, kernel_size=3,
                          stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(in_ch, affine=True),  # Replaces LADE_D
            nn.LeakyReLU(0.2, inplace=True),

            # Conv Stride 1 (Increase channel here)
            spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=3,
                          stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(out_ch, affine=True),  # Replaces LADE_D
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x
