import torch
import torch.nn as nn

class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()
        # L1 Loss cho kênh Y (Độ sáng)
        self.l1 = nn.L1Loss()
        # Huber Loss (Smooth L1 Loss trong PyTorch) cho kênh U và V (Màu sắc)
        # Mặc định beta=1.0 tương đương với delta=1.0 trong tf.losses.huber_loss
        self.huber = nn.SmoothL1Loss() 

        # Ma trận chuyển đổi RGB -> YUV (Chuẩn BT.601 của tf.image.rgb_to_yuv)
        matrix = torch.tensor([
            [ 0.299,       0.587,       0.114],
            [-0.14714119, -0.28886916,  0.43601035],
            [ 0.61497538, -0.51496512, -0.10001026]
        ], dtype=torch.float32)
        
        # Đăng ký vào buffer để biến này đi theo model lên GPU
        self.register_buffer('rgb2yuv_matrix', matrix)

    def rgb_to_yuv(self, image):
        """
        image shape: [Batch, 3, H, W] (Đã được scale về dải [0, 1])
        """
        # Đổi shape từ [B, 3, H, W] -> [B, H, W, 3] để nhân ma trận
        img_permuted = image.permute(0, 2, 3, 1)
        
        # Thực hiện phép nhân ma trận: (RGB) x (Matrix^T)
        yuv = torch.matmul(img_permuted, self.rgb2yuv_matrix.T)
        
        # Trả lại shape [B, 3, H, W]
        return yuv.permute(0, 3, 1, 2)

    def forward(self, fake_img, real_photo):
        """
        fake_img: Ảnh Generator sinh ra
        real_photo: Ảnh chụp thực tế (giữ vai trò mỏ neo màu sắc)
        """
        # 1. Ép dải pixel từ [-1, 1] về [0, 1] giống code tf: (rgb + 1.0)/2.0
        fake_img_01 = (fake_img + 1.0) / 2.0
        real_photo_01 = (real_photo + 1.0) / 2.0

        # 2. Chuyển đổi sang không gian màu YUV
        fake_yuv = self.rgb_to_yuv(fake_img_01)
        
        with torch.no_grad():
            real_yuv = self.rgb_to_yuv(real_photo_01)

        # 3. Tách riêng các kênh Y, U, V (index 0, 1, 2)
        fake_y, fake_u, fake_v = fake_yuv[:, 0:1, :, :], fake_yuv[:, 1:2, :, :], fake_yuv[:, 2:3, :, :]
        real_y, real_u, real_v = real_yuv[:, 0:1, :, :], real_yuv[:, 1:2, :, :], real_yuv[:, 2:3, :, :]

        # 4. Tính toán Loss kết hợp (L1 cho Y, Huber cho U và V)
        loss_y = self.l1(fake_y, real_y)
        loss_u = self.huber(fake_u, real_u)
        loss_v = self.huber(fake_v, real_v)

        # Tổng hợp Loss
        return loss_y + loss_u + loss_v