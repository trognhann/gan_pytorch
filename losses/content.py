import torch
import torch.nn as nn
from .vgg import VGG19


class ContentLoss(nn.Module):
    def __init__(self, vgg=None):
        super(ContentLoss, self).__init__()
        self.vgg = vgg if vgg is not None else VGG19()
        self.l1 = nn.L1Loss()

    def forward(self, real, fake):
        _, _, real_feat = self.vgg(real)
        _, _, fake_feat = self.vgg(fake)
        # nn.L1Loss(reduction='mean') already averages over all B×C×H×W elements.
        # Previously divided by C=512 again (from TF reduce_mean misinterpretation),
        # which made content loss ~512x too small vs style/color losses.
        loss = self.l1(real_feat, fake_feat)
        return loss
