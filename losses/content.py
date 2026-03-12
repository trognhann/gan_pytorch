import torch
import torch.nn as nn
from .vgg import VGG19


class ContentLoss(nn.Module):
    def __init__(self, vgg=None):
        super(ContentLoss, self).__init__()
        self.vgg = vgg if vgg is not None else VGG19()
        self.l1 = nn.L1Loss()

    def forward(self, real_photo, fake_img):
        _, _, real_f4 = self.vgg(real_photo)
        _, _, fake_f4 = self.vgg(fake_img)

        c = real_f4.size(1)
        loss = self.l1(fake_f4, real_f4.detach()) / c

        return loss
