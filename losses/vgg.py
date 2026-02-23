import torch
import torch.nn as nn
import torchvision.models as models


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()

        for x in range(8):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(8, 17):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(17, 26):
            self.slice3.add_module(str(x), vgg[x])

        for param in self.parameters():
            param.requires_grad = False

        # Normalization constants for PyTorch VGG
        self.register_buffer("mean", torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # x is assumed to be in range [-1, 1]
        x = (x + 1.0) / 2.0
        x = (x - self.mean) / self.std

        h_relu2_2 = self.slice1(x)
        h_relu3_3 = self.slice2(h_relu2_2)
        h_relu4_4 = self.slice3(h_relu3_3)
        return h_relu2_2, h_relu3_3, h_relu4_4
