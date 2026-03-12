import torch
import torch.nn as nn
import torch.nn.functional as F


class ExternalAttention_v3(nn.Module):
    def __init__(self, c, k=128):
        super(ExternalAttention_v3, self).__init__()
        self.k = k
        self.conv1 = nn.Conv2d(c, c, kernel_size=1,
                               stride=1, padding=0, bias=True)

        # mk shape: [c, k]. Initialized orthogonally or xavier.
        self.mk = nn.Parameter(torch.empty(c, k))
        nn.init.xavier_uniform_(self.mk)

        self.conv2 = nn.Conv2d(c, c, kernel_size=1,
                               stride=1, padding=0, bias=True)

        # Original TF uses batch_norm_wrapper(is_training)
        # By default momentum=0.1
        self.bn = nn.BatchNorm2d(c, momentum=0.1, eps=1e-3)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        idn = x
        b, c, h, w = x.shape

        out = self.conv1(x)  # [B, c, H, W]
        out = out.view(b, c, -1).transpose(1, 2)  # [B, HW, c]

        # First 1D Conv equivalent (matmul with mk)
        attn = torch.matmul(out, self.mk)  # [B, HW, c] x [c, k] -> [B, HW, k]

        # Softmax over axis=1 (HW)
        attn = F.softmax(attn, dim=1)
        # Normalize over axis=2 (k)
        attn = attn / (1e-5 + attn.sum(dim=2, keepdim=True))

        # Second 1D Conv equivalent (matmul with transposed mk)
        # [B, HW, k] x [k, c] -> [B, HW, c]
        out = torch.matmul(attn, self.mk.t())

        # Reshape back to image format
        out = out.transpose(1, 2).view(b, c, h, w)  # [B, c, H, W]

        out = self.conv2(out)
        out = self.bn(out)

        out = out + idn
        return self.lrelu(out)
