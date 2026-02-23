import torch
import torch.nn as nn
from .vgg import VGG19


class RegionSmoothingLoss(nn.Module):
    def __init__(self):
        super(RegionSmoothingLoss, self).__init__()
        self.vgg = VGG19()
        self.l1 = nn.L1Loss()

    def forward(self, seg, fake, weight):
        _, _, seg_feat = self.vgg(seg)
        _, _, fake_feat = self.vgg(fake)
        c = seg_feat.size(1)
        loss = self.l1(seg_feat, fake_feat) / c
        return weight * loss
