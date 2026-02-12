import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h*w)
        return G


class GuidedFilter(nn.Module):
    def __init__(self, r=1, eps=1e-2):
        super(GuidedFilter, self).__init__()
        self.r = r
        self.eps = eps
        self.box = nn.AvgPool2d(
            2*r+1, stride=1, padding=r, count_include_pad=False)

    def forward(self, x, y):
        # x: guidance, y: filtering input
        # If guidance is same as input, self-guided

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

        # RGB to XYZ Matrix
        self.rgb_to_xyz_matrix = torch.tensor([
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227]
        ]).float()

    def rgb_to_lab(self, img):
        # img: [-1, 1] -> [0, 1]
        img = (img + 1) / 2

        device = img.device
        matrix = self.rgb_to_xyz_matrix.to(device)

        # RGB to XYZ
        # (B, 3, H, W) -> (B, H, W, 3)
        img_perm = img.permute(0, 2, 3, 1)
        xyz = torch.matmul(img_perm, matrix.t())

        # Normalize XYZ for Lab
        # Reference white (D65)
        # Xn, Yn, Zn = 0.950456, 1.0, 1.088754
        xyz = xyz / torch.tensor([0.950456, 1.0, 1.088754]).to(device)

        # f(t) function
        # t > 0.008856 -> t^(1/3)
        # t <= 0.008856 -> 7.787*t + 16/116
        mask = xyz > 0.008856
        f_xyz = torch.where(mask, torch.pow(xyz, 1/3), 7.787 * xyz + 16/116)

        # Lab
        L = 116 * f_xyz[..., 1] - 16
        a = 500 * (f_xyz[..., 0] - f_xyz[..., 1])
        b = 200 * (f_xyz[..., 1] - f_xyz[..., 2])

        # Stack back (B, H, W, 3) -> (B, 3, H, W)
        lab = torch.stack([L, a, b], dim=-1).permute(0, 3, 1, 2)
        return lab

    def forward(self, x, y):
        return self.l1(self.rgb_to_lab(x), self.rgb_to_lab(y))


class VGG19(nn.Module):
    def __init__(self, vgg_path):
        super(VGG19, self).__init__()

        # Manual load from .npy
        if not os.path.exists(vgg_path):
            # Fallback to torchvision? User said 'High Priority' to fix, 'Low' to replace.
            # I will try to load standard if NPY not found to be safe.
            import torchvision.models as models
            print("Loading torchvision VGG19...")
            vgg = models.vgg19(pretrained=True).features
            self.from_torchvision = True
            self.model = vgg
            return

        self.from_torchvision = False
        data = np.load(vgg_path, allow_pickle=True, encoding='latin1').item()

        def make_layer(name, in_c, out_c):
            w, b = data[name]
            w = np.transpose(w, (3, 2, 0, 1))
            conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
            conv.weight.data.copy_(torch.from_numpy(w))
            conv.bias.data.copy_(torch.from_numpy(b))
            return conv

        self.conv1_1 = make_layer('conv1_1', 3, 64)
        self.conv1_2 = make_layer('conv1_2', 64, 64)
        self.conv2_1 = make_layer('conv2_1', 64, 128)
        self.conv2_2 = make_layer('conv2_2', 128, 128)
        self.conv3_1 = make_layer('conv3_1', 128, 256)
        self.conv3_2 = make_layer('conv3_2', 256, 256)
        self.conv3_3 = make_layer('conv3_3', 256, 256)
        self.conv3_4 = make_layer('conv3_4', 256, 256)
        self.conv4_1 = make_layer('conv4_1', 256, 512)
        self.conv4_2 = make_layer('conv4_2', 512, 512)
        self.conv4_3 = make_layer('conv4_3', 512, 512)
        self.conv4_4 = make_layer('conv4_4', 512, 512)  # Content
        self.conv5_1 = make_layer('conv5_1', 512, 512)
        self.conv5_2 = make_layer('conv5_2', 512, 512)
        self.conv5_3 = make_layer('conv5_3', 512, 512)
        self.conv5_4 = make_layer('conv5_4', 512, 512)

    def forward(self, x):
        # Expecting [0, 1] RGB input? NO, VGG usually expects [0, 255] BGR - Mean.
        # But this is specific NPY from AnimeGAN repo presumably trained on RGB [-1, 1] or [0, 255]?
        # Usually AnimeGAN uses BGR [0, 255] minus mean.
        # Let's normalize inside logic if possible.
        # Assuming input x is [-1, 1].

        out = {}
        if self.from_torchvision:
            # Standard VGG logic
            # Normalize [-1, 1] -> [0, 1] -> Mean/Std
            x = (x + 1) / 2
            mean = torch.tensor([0.485, 0.456, 0.406]).to(
                x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).to(
                x.device).view(1, 3, 1, 1)
            x = (x - mean) / std
            # This returns last layer? No, self.model is Sequential.
            features = self.model(x)
            # Need to extract intermediate...
            # For now, let's just implement the manual one properly.
            # Placeholder for torchvision fallback if needed.
            pass
        else:
            # Manual NPY
            # Preprocess: [-1, 1] -> [0, 255] BGR - Mean?
            # Custom AnimeGAN VGG implementation typically takes RGB [0, 255] and subtracts VGG mean.
            # Mean: [103.939, 116.779, 123.68] BGR
            # Let's do BGR conversion here.

            x = (x + 1) * 127.5

            # RGB to BGR
            x = x[:, [2, 1, 0], :, :]

            # Subtract Mean
            mean = torch.tensor([103.939, 116.779, 123.68]).to(
                x.device).view(1, 3, 1, 1)
            x = x - mean

            x = F.relu(self.conv1_1(x))
            x = F.relu(self.conv1_2(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2_1(x))
            x = F.relu(self.conv2_2(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv3_1(x))
            x = F.relu(self.conv3_2(x))
            x = F.relu(self.conv3_3(x))
            x = F.relu(self.conv3_4(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv4_1(x))
            x = F.relu(self.conv4_2(x))
            x = F.relu(self.conv4_3(x))
            x = F.relu(self.conv4_4(x))
            out['relu4_4'] = x
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv5_1(x))
            x = F.relu(self.conv5_2(x))
            x = F.relu(self.conv5_3(x))
            x = F.relu(self.conv5_4(x))
            out['relu5_4'] = x

        return out


class AnimeGANLoss(nn.Module):
    def __init__(self, device='cpu', vgg_path='vgg19_weight/vgg19_no_fc.npy'):
        super(AnimeGANLoss, self).__init__()
        self.vgg = VGG19(vgg_path).to(device)
        self.vgg.eval()

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.gram = GramMatrix()
        self.guided_filter = GuidedFilter()

    def to_gray(self, x):
        # RGB to Grayscale: Y = 0.299 R + 0.587 G + 0.114 B
        # Assume x is [-1, 1]
        x = (x + 1) / 2
        gray = 0.299 * x[:, 0, :, :] + 0.587 * \
            x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
        gray = gray.unsqueeze(1)  # (B, 1, H, W)
        # Duplicate channels to 3 for VGG
        gray = gray.repeat(1, 3, 1, 1)
        # Scale back to [-1, 1]
        gray = (gray * 2) - 1
        return gray

    def content_loss(self, fake, real):
        f_fake = self.vgg(fake)['relu4_4']
        f_real = self.vgg(real)['relu4_4']
        return self.l1_loss(f_fake, f_real)

    def style_loss(self, fake, style):
        # Grayscale style loss
        fake_gray = self.to_gray(fake)
        style_gray = self.to_gray(style)

        f_fake = self.vgg(fake_gray)['relu4_4']
        f_style = self.vgg(style_gray)['relu4_4']

        gram_fake = self.gram(f_fake)
        gram_style = self.gram(f_style)
        return self.l1_loss(gram_fake, gram_style)

    def gram_loss(self, fake, style):
        return self.style_loss(fake, style)

    def variation_loss(self, x):
        return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
            torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))

    def adversarial_loss_lsgan(self, discriminator, fake_img, real_label=True):
        pred = discriminator(fake_img)
        target = torch.ones_like(
            pred) if real_label else torch.zeros_like(pred)
        return self.mse_loss(pred, target)
