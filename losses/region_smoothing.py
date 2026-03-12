import torch
import torch.nn as nn
from .vgg import VGG19

class RegionSmoothingLoss(nn.Module):
    def __init__(self, weight=1.0, vgg=None):
        super(RegionSmoothingLoss, self).__init__()
        self.vgg = vgg if vgg is not None else VGG19()
        self.weight = weight
        self.l1 = nn.L1Loss()

    def forward(self, superpixel_photo, fake_img):
        _, _, seg_f4 = self.vgg(superpixel_photo)
        _, _, fake_f4 = self.vgg(fake_img)

        c = seg_f4.size(1)

        loss = self.weight * (self.l1(fake_f4, seg_f4.detach()) / c)

        return loss