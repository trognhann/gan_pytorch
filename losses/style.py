import torch
import torch.nn as nn
from .vgg import VGG19


class StyleLoss(nn.Module):
    def __init__(self, weights=[0.1, 5.0, 25.0]):
        super(StyleLoss, self).__init__()
        self.vgg = VGG19()
        self.weights = weights
        self.l1 = nn.L1Loss()

    def gram_matrix(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        G = torch.bmm(features, features.transpose(1, 2))
        return G / (c * h * w)

    def forward(self, style, fake):
        s2, s3, s4 = self.vgg(style)
        f2, f3, f4 = self.vgg(fake)

        s2 = s2 - s2.mean(dim=[2, 3], keepdim=True)
        f2 = f2 - f2.mean(dim=[2, 3], keepdim=True)

        s3 = s3 - s3.mean(dim=[2, 3], keepdim=True)
        f3 = f3 - f3.mean(dim=[2, 3], keepdim=True)

        s4 = s4 - s4.mean(dim=[2, 3], keepdim=True)
        f4 = f4 - f4.mean(dim=[2, 3], keepdim=True)

        l2 = self.weights[0] * self.l1(self.gram_matrix(s2),
                                       self.gram_matrix(f2)) / s2.size(1)
        l3 = self.weights[1] * self.l1(self.gram_matrix(s3),
                                       self.gram_matrix(f3)) / s3.size(1)
        l4 = self.weights[2] * self.l1(self.gram_matrix(s4),
                                       self.gram_matrix(f4)) / s4.size(1)

        return l2 + l3 + l4
