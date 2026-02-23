import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()

    def d_support_loss(self, anime_gray_logit, fake_gray_logit, anime_smooth_logit):
        real_loss = 0.5 * torch.mean((anime_gray_logit - 0.9) ** 2)
        fake_loss = 1.0 * torch.mean((fake_gray_logit - 0.1) ** 2)
        smooth_loss = 2.0 * torch.mean((anime_smooth_logit - 0.1) ** 2)
        return real_loss + fake_loss + smooth_loss

    def g_support_adv_loss(self, fake_gray_logit):
        return torch.mean((fake_gray_logit - 0.9) ** 2)

    def d_main_loss(self, real_logit, fake_logit):
        real_loss = torch.mean((real_logit - 1.0) ** 2)
        # LSGAN with targets 1 and 0 for discriminator_m
        fake_loss = torch.mean(fake_logit ** 2)
        return real_loss + fake_loss

    def g_main_adv_loss(self, fake_logit):
        return torch.mean((fake_logit - 1.0) ** 2)

    def tv_loss(self, x):
        count_h = x.numel()
        count_w = x.numel()

        dh = x[:, :, 1:, :] - x[:, :, :-1, :]
        dw = x[:, :, :, 1:] - x[:, :, :, :-1]

        tv = 0.5 * torch.sum(dh ** 2) / count_h + 0.5 * \
            torch.sum(dw ** 2) / count_w
        return tv
