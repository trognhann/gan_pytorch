import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h*w)
        return G


class GuidedFilter(nn.Module):
    def __init__(self, r=1, eps=5e-3):
        super(GuidedFilter, self).__init__()
        self.r = r
        self.eps = eps
        self.box = nn.AvgPool2d(
            2*r+1, stride=1, padding=r, count_include_pad=False)

    def forward(self, x, y):
        # x: guidance, y: filtering input
        N, C, H, W = x.size()
        mean_x = self.box(x)
        mean_y = self.box(y)
        cov_xy = self.box(x * y) - mean_x * mean_y
        var_x = self.box(x * x) - mean_x * mean_x
        A = cov_xy / (var_x + self.eps)
        b = mean_y - A * mean_x
        mean_A = self.box(A)
        mean_b = self.box(b)
        output = mean_A * x + mean_b
        return output


class AnimeGANLoss(nn.Module):
    def __init__(self, device='cpu'):
        super(AnimeGANLoss, self).__init__()
        self.device = device

        # Use torchvision VGG19
        vgg = models.vgg19(weights='DEFAULT').features
        vgg.eval()
        self.vgg = vgg.to(device)

        # Standard VGG Mean/Std for normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(
            1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(
            1, 3, 1, 1).to(device)

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.gram = GramMatrix()

    def normalize_vgg(self, x):
        # Input: [-1, 1] RGB
        x = (x + 1) * 0.5
        return (x - self.mean) / self.std

    def extract_features(self, x):
        x = self.normalize_vgg(x)
        features = {}
        out = x
        for i, layer in enumerate(self.vgg):
            out = layer(out)
            if i == 26:  # relu4_4
                features['relu4_4'] = out
                break
        return features

    def rgb_to_gray(self, img):
        # ITU-R BT.601 formula
        # Input: [-1, 1] RGB
        # Output: [-1, 1] Gray (B, 1, H, W)
        x_norm = (img + 1) * 0.5
        y = 0.299 * x_norm[:, 0, :, :] + 0.587 * \
            x_norm[:, 1, :, :] + 0.114 * x_norm[:, 2, :, :]
        y = y.unsqueeze(1)
        return (y * 2) - 1

    def rgb_to_yuv(self, img):
        # Input: [-1, 1] RGB
        # Output: [-1, 1] YUV
        # Using standard RGB to YUV matrix
        img = (img + 1) * 0.5  # [0, 1]

        r, g, b = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.147 * r - 0.289 * g + 0.436 * b
        v = 0.615 * r - 0.515 * g - 0.100 * b

        yuv = torch.stack([y, u, v], dim=1)
        # Scale back? L1 loss is relative, but usually we keep ranges similar.
        # Y is [0, 1]. U is [-0.436, 0.436]. V is [-0.615, 0.615].
        return yuv

    def content_loss(self, fake, real):
        f_fake = self.extract_features(fake)['relu4_4']
        f_real = self.extract_features(real)['relu4_4']
        return self.l1_loss(f_fake, f_real)

    def style_loss(self, fake, style):
        # Grayscale Style Loss
        # Convert to Gray and then expand to 3 channels for VGG
        fake_gray = self.rgb_to_gray(fake).repeat(1, 3, 1, 1)
        style_gray = self.rgb_to_gray(style).repeat(1, 3, 1, 1)

        f_fake = self.extract_features(fake_gray)['relu4_4']
        f_style = self.extract_features(style_gray)['relu4_4']

        gram_fake = self.gram(f_fake)
        gram_style = self.gram(f_style)
        return self.l1_loss(gram_fake, gram_style)

    def color_loss(self, x, y):
        # YUV Color Loss (better than RGB, simpler fallback for Lab)
        return self.l1_loss(self.rgb_to_yuv(x), self.rgb_to_yuv(y))

    def variation_loss(self, x):
        return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
            torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))

    def adversarial_loss_lsgan(self, discriminator, fake_img, real_label=True):
        pred = discriminator(fake_img)
        target = torch.ones_like(
            pred) if real_label else torch.zeros_like(pred)
        return self.mse_loss(pred, target)
