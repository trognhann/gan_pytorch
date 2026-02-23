import torch
import torch.nn as nn
from .vgg import VGG19


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.vgg = VGG19()
        self.l1 = nn.L1Loss()

    def forward(self, real, fake):
        _, _, real_feat = self.vgg(real)
        _, _, fake_feat = self.vgg(fake)
        # Divide by feature channels mapped from TF reduce_mean
        c = real_feat.size(1)
        loss = self.l1(real_feat, fake_feat) / c
        return loss
