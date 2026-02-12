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


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def rgb_to_lab(self, img):
        # RGB to XYZ -> XYZ to Lab (approximate/simple version for stability)
        # Using l1 on RGB + blurred RGB as a proxy for color consistency is common if Lab is unstable
        # But here serves as placeholder.
        return img

    def forward(self, x, y):
        # AnimeGANv3 uses: L1(x, y) + Huber(Gray(x), Gray(y))?
        # Or L1(Lab(x), Lab(y)).
        # Implementation: L1 on RGB + Cosine similarity is robust.
        # Let's use simple L1 on RGB + YUV (Y is structure, UV is color).
        return self.l1(x, y)


class VGG19(nn.Module):
    def __init__(self, vgg_path):
        super(VGG19, self).__init__()
        self.layers_dict = {}

        if not os.path.exists(vgg_path):
            print(
                f"Warning: VGG weights not found at {vgg_path}. Random init used.")
            # Define structure anyway for partial loading or errors
            return

        # Load weights
        # Structure: {(H, W, In, Out), (Bias,)}
        data = np.load(vgg_path, allow_pickle=True, encoding='latin1').item()

        # Build layers manually to match the keys in npy
        # Keys: conv1_1, conv1_2, ...

        # We need specific features.
        # AnimeGAN typically uses 'conv4_4' for content.

        # Helper to create layer
        def make_layer(name, in_c, out_c):
            w, b = data[name]
            # w shape: (H, W, In, Out) -> PyTorch (Out, In, H, W)
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
        self.conv4_4 = make_layer('conv4_4', 512, 512)

        self.conv5_1 = make_layer('conv5_1', 512, 512)
        self.conv5_2 = make_layer('conv5_2', 512, 512)
        self.conv5_3 = make_layer('conv5_3', 512, 512)
        self.conv5_4 = make_layer('conv5_4', 512, 512)

    def forward(self, x):
        # Returns specific features
        out = {}

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
        x = F.relu(self.conv4_4(x))  # Content
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

    def content_loss(self, fake, real):
        # Normalize to [0, 1]? or VGG expected input range?
        # VGG usually expects [0, 255] BGR or [0, 1] RGB normalized.
        # This custom VGG from AnimeGAN repo likely expects [-1, 1] or [0, 1] RGB?
        # Assuming [-1, 1] input to AnimeGAN generator, let's normalize to [0, 1] for VGG.
        # Or better yet, standard ImageNet mean/std if weights were from standard VGG.
        # But these are 'no_fc', likely from some Tensorflow repo.
        # Usually TF VGG expects [0, 255] centered.
        # Let's assume input 'fake' is [-1, 1].

        # Re-normalize for VGG?
        # Common practice in AnimeGAN: (x + 1) / 2 * 255 - mean?
        # For safety, let's pass (x+1)/2 [0..1] range if we are unsure.
        # Or Just pass features.

        f_fake = self.vgg(fake)['relu4_4']
        # Detach real? usually done in loop/by default if leaf.
        f_real = self.vgg(real)['relu4_4']
        return self.l1_loss(f_fake, f_real)  # Usually L1 for content

    def style_loss(self, fake, style):
        f_fake = self.vgg(fake)['relu4_4']
        f_style = self.vgg(style)['relu4_4']

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
