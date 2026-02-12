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


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def rgb_to_lab(self, img):
        # Differentiable RGB to Lab
        # Input img: [-1, 1] RGB
        img = (img + 1) * 0.5  # [0, 1]

        # RGB to XYZ
        # Matrices from Kornia/standard definitions
        r, g, b = img[:, 0, ...], img[:, 1, ...], img[:, 2, ...]

        # Linearize RGB (assuming sRGB input)
        # x = torch.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
        # Simplified linear approximation can be used for stability, but let's try standard.
        # For GANs, often a linear approximation is sufficient and more stable for gradients.

        X = 0.412453 * r + 0.357580 * g + 0.180423 * b
        Y = 0.212671 * r + 0.715160 * g + 0.072169 * b
        Z = 0.019334 * r + 0.119193 * g + 0.950227 * b

        # XYZ to Lab
        # Reference White D65
        Xn, Yn, Zn = 0.95047, 1.00000, 1.08883

        X = X / Xn
        Y = Y / Yn
        Z = Z / Zn

        # f(t)
        # t > 0.008856 -> t^(1/3)
        # t <= 0.008856 -> 7.787*t + 16/116
        threshold = 0.008856

        def f(t):
            return torch.where(t > threshold, torch.pow(torch.clamp(t, min=threshold), 1/3), 7.787 * t + 16/116)

        L = 116 * f(Y) - 16
        a = 500 * (f(X) - f(Y))
        b = 200 * (f(Y) - f(Z))

        return torch.stack([L, a, b], dim=1)

    def forward(self, x, y):
        return self.l1(self.rgb_to_lab(x), self.rgb_to_lab(y))


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
        # VGG19 Layer mapping (approx)
        # conv4_4 is what we usually want for content.
        # relu4_4 is at index 26

        out = x
        for i, layer in enumerate(self.vgg):
            out = layer(out)
            if i == 26:  # relu4_4
                features['relu4_4'] = out
                break
                # Can extend for style features if needed (relu1_1, relu2_1, etc.)
                # But typically efficient Style Loss uses single layer or just multiple calls.
        return features

    def rgb_to_gray(self, x):
        # Input: [-1, 1] RGB
        # Y = 0.299R + 0.587G + 0.114B
        # Output: [-1, 1] Gray (3 channels)
        x_norm = (x + 1) * 0.5
        gray = 0.299 * x_norm[:, 0, ...] + 0.587 * \
            x_norm[:, 1, ...] + 0.114 * x_norm[:, 2, ...]
        gray = gray.unsqueeze(1).repeat(1, 3, 1, 1)  # (B, 3, H, W)
        return (gray - 0.5) * 2

    def content_loss(self, fake, real):
        # Standard content loss (RGB)
        f_fake = self.extract_features(fake)['relu4_4']
        f_real = self.extract_features(real)['relu4_4']
        return self.l1_loss(f_fake, f_real)

    def style_loss(self, fake, style):
        # Grayscale Style Loss - Crucial for structure only
        fake_gray = self.rgb_to_gray(fake)
        style_gray = self.rgb_to_gray(style)

        # For Gram matrix, we usually use multiple layers.
        # But if restricted to one or similar to content:
        f_fake = self.extract_features(fake_gray)['relu4_4']
        f_style = self.extract_features(style_gray)['relu4_4']

        gram_fake = self.gram(f_fake)
        gram_style = self.gram(f_style)
        return self.l1_loss(gram_fake, gram_style)

    def variation_loss(self, x):
        return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
            torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))

    def adversarial_loss_lsgan(self, discriminator, fake_img, real_label=True):
        pred = discriminator(fake_img)
        target = torch.ones_like(
            pred) if real_label else torch.zeros_like(pred)
        return self.mse_loss(pred, target)
