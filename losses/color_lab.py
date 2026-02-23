import torch
import torch.nn as nn


def rgb_to_xyz(input):
    matrix = torch.tensor([
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227]
    ], dtype=input.dtype, device=input.device)

    mask = input > 0.04045
    input_lin = torch.where(mask, torch.pow(
        (input + 0.055) / 1.055, 2.4), input / 12.92)

    input_lin = input_lin.permute(0, 2, 3, 1)  # B, H, W, C
    xyz = torch.matmul(input_lin, matrix.t())
    xyz = xyz.permute(0, 3, 1, 2)  # B, C, H, W
    return xyz


def xyz_to_lab(xyz):
    illuminant = torch.tensor(
        [0.95047, 1.0, 1.08883], dtype=xyz.dtype, device=xyz.device).view(1, 3, 1, 1)
    xyz = xyz / illuminant
    mask = xyz > 0.008856
    f = torch.where(mask, torch.pow(xyz, 1.0/3.0), xyz * 7.787 + 16.0/116.0)
    l = (f[:, 1:2, :, :] * 116.0) - 16.0
    a = (f[:, 0:1, :, :] - f[:, 1:2, :, :]) * 500.0
    b = (f[:, 1:2, :, :] - f[:, 2:3, :, :]) * 200.0
    return torch.cat([l, a, b], dim=1)


def rgb_to_lab(rgb):
    return xyz_to_lab(rgb_to_xyz(rgb))


class ColorLoss(nn.Module):
    def __init__(self, weight=10.0):
        super(ColorLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.weight = weight

    def forward(self, photo, fake):
        photo = (photo + 1.0) / 2.0
        fake = (fake + 1.0) / 2.0

        photo_lab = rgb_to_lab(photo)
        fake_lab = rgb_to_lab(fake)

        photo_l = photo_lab[:, 0:1] / 100.0
        fake_l = fake_lab[:, 0:1] / 100.0

        photo_a = (photo_lab[:, 1:2] + 128.0) / 255.0
        fake_a = (fake_lab[:, 1:2] + 128.0) / 255.0

        photo_b = (photo_lab[:, 2:3] + 128.0) / 255.0
        fake_b = (fake_lab[:, 2:3] + 128.0) / 255.0

        loss = 2.0 * self.l1(photo_l, fake_l) + \
            self.l1(photo_a, fake_a) + self.l1(photo_b, fake_b)
        return self.weight * loss
