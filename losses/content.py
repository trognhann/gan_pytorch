import torch
import torch.nn as nn


class ContentLoss(nn.Module):
    def __init__(self, weight=1.0, vgg=None, feature_extractor=None):
        """
        Args:
            weight: Loss weight multiplier.
            vgg: (deprecated) VGG19 instance for backward compat.
            feature_extractor: Any feature extractor (VGG19 or CLIP) that
                               returns (f_low, f_mid, f_high) tuple.
        """
        super(ContentLoss, self).__init__()
        # Prefer feature_extractor; fall back to vgg for backward compat
        if feature_extractor is not None:
            self.extractor = feature_extractor
        elif vgg is not None:
            self.extractor = vgg
        else:
            from .vgg import VGG19
            self.extractor = VGG19()
        self.weight = weight
        self.l1 = nn.L1Loss()

    def forward(self, real_photo, fake_img):
        _, _, real_f4 = self.extractor(real_photo)
        _, _, fake_f4 = self.extractor(fake_img)

        c = real_f4.size(1)
        loss = self.weight * (self.l1(fake_f4, real_f4.detach()) / c)

        return loss
