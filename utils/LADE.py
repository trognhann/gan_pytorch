import torch
import torch.nn as nn
import torch.nn.functional as F

class LADE(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super(LADE, self).__init__()
        self.eps = eps
        # Tích chập từng điểm (Pointwise Convolution) để tính trọng số tuyến tính
        self.pointwise_conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # 1. Chuẩn hóa đặc trưng (Instance Normalization gốc)
        x_norm = F.instance_norm(x, eps=self.eps)
        
        # 2. Áp dụng Pointwise Conv để tạo ra tensor P(x)
        p_x = self.pointwise_conv(x)
        
        # 3. Tính Mean (beta) và Std (gamma) trên không gian H, W của tensor P(x)
        # Khớp với Công thức 4 và 5 trong bài báo
        beta = torch.mean(p_x, dim=[7, 8], keepdim=True)
        gamma = torch.std(p_x, dim=[7, 8], unbiased=False, keepdim=True) + self.eps
        
        # 4. Giải chuẩn hóa thích ứng tuyến tính
        out = x_norm * gamma + beta
        return out