import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm


class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=32):
        super(Discriminator, self).__init__()

        # Use Spectral Normalization for stability
        self.model = nn.Sequential(
            # Layer 1
            spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=3,
                          stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, True),

            # Layer 2: Strided
            spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=3,
                          stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 2, kernel_size=3, stride=1,
                      padding=1, bias=False),  # Grouped conv-like block usually
            nn.LeakyReLU(0.2, True),

            # Layer 3: Strided
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3,
                          stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 4, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),

            # Layer 4
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 4, kernel_size=3,
                          stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, True),

            # Output Layer (PatchGAN output, 1 channel)
            nn.Conv2d(ndf * 4, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)
