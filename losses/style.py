import torch
import torch.nn as nn
from .vgg import VGG19


class StyleLoss(nn.Module):
    def __init__(self, weights=[0.1, 5.0, 25.0], vgg=None):
        super(StyleLoss, self).__init__()
        self.vgg = vgg if vgg is not None else VGG19()
        self.weights = weights
        self.l1 = nn.L1Loss()

    def rgb_to_grayscale(self, x):
        gray = 0.2125 * x[:, 0:1, :, :] + 0.7154 * \
            x[:, 1:2, :, :] + 0.0721 * x[:, 2:3, :, :]
        return gray.repeat(1, 3, 1, 1)

    def gram_matrix(self, x):
        b, c, h, w = x.size()
        with torch.amp.autocast('cuda', enabled=False):
            features = x.view(b, c, h * w).to(torch.float32)
            G = torch.bmm(features, features.transpose(1, 2))
        return G / (c * h * w)

    def forward(self, style, fake):
        style_gray = self.rgb_to_grayscale(style)
        fake_gray = self.rgb_to_grayscale(fake)

        s2, s3, s4 = self.vgg(style_gray)
        f2, f3, f4 = self.vgg(fake_gray)

        with torch.amp.autocast('cuda', enabled=False):
            s2, f2 = s2.to(torch.float32), f2.to(torch.float32)
            s3, f3 = s3.to(torch.float32), f3.to(torch.float32)
            s4, f4 = s4.to(torch.float32), f4.to(torch.float32)

            s2 = s2 - s2.mean(dim=[2, 3], keepdim=True)
            f2 = f2 - f2.mean(dim=[2, 3], keepdim=True)

            s3 = s3 - s3.mean(dim=[2, 3], keepdim=True)
            f3 = f3 - f3.mean(dim=[2, 3], keepdim=True)

            s4 = s4 - s4.mean(dim=[2, 3], keepdim=True)
            f4 = f4 - f4.mean(dim=[2, 3], keepdim=True)

            c2 = s2.size(1)
            c3 = s3.size(1)
            c4 = s4.size(1)

            l2 = self.weights[0] * \
                (self.l1(self.gram_matrix(s2), self.gram_matrix(f2)) / c2)
            l3 = self.weights[1] * \
                (self.l1(self.gram_matrix(s3), self.gram_matrix(f3)) / c3)
            l4 = self.weights[2] * \
                (self.l1(self.gram_matrix(s4), self.gram_matrix(f4)) / c4)

        return l2 + l3 + l4
