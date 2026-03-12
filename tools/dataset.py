import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image

class AnimeDataset(Dataset):
    def __init__(self, root_dir, img_size=256):
        self.root_dir = root_dir
        self.img_size = img_size

        # Directories
        self.photo_dir = os.path.join(root_dir, 'train_photo')
        self.anime_dir = os.path.join(root_dir, 'style')
        self.smooth_dir = os.path.join(root_dir, 'style_smooth_noise')
        self.photo_smooth_dir = os.path.join(root_dir, 'train_photo_superpixel')

        # File lists (Chỉ cần lấy danh sách gốc làm mỏ neo)
        self.photos = self._load_files(self.photo_dir)
        self.animes = self._load_files(self.anime_dir)

        self.len_photo = len(self.photos)
        self.len_anime = len(self.animes)

    def _load_files(self, dir_path):
        if not os.path.exists(dir_path):
            return []
        files = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(
            ('.jpg', '.jpeg', '.png', '.bmp'))])
        return files

    def __len__(self):
        return max(self.len_photo, self.len_anime)

    def paired_transform(self, img_a, img_b):
        """Đảm bảo hai ảnh được transform giống hệt nhau về mặt không gian"""
        # Resize
        img_a = TF.resize(img_a, (self.img_size, self.img_size))
        img_b = TF.resize(img_b, (self.img_size, self.img_size))
        
        # Random Horizontal Flip (Đồng bộ)
        if random.random() > 0.5:
            img_a = TF.hflip(img_a)
            img_b = TF.hflip(img_b)
            
        # To Tensor & Normalize [-1, 1]
        img_a = TF.normalize(TF.to_tensor(img_a), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        img_b = TF.normalize(TF.to_tensor(img_b), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        return img_a, img_b

    def _read_img_pil(self, path):
        """Đọc bằng PIL để tương thích tốt nhất với torchvision.transforms.functional"""
        try:
            return Image.open(path).convert('RGB')
        except Exception:
            # Fallback tạo ảnh đen nếu lỗi file để tránh crash DataLoader
            return Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))

    def __getitem__(self, idx):
        # 1. Lấy tên file chuẩn xác
        photo_name = self.photos[idx % self.len_photo]
        # Lấy ngẫu nhiên ảnh Anime để tăng tính đa dạng (hoặc dùng modulo cũng được)
        anime_name = self.animes[random.randint(0, self.len_anime - 1)]

        # 2. Setup đường dẫn (Ép cặp photo-superpixel và anime-smooth phải trùng tên file)
        photo_path = os.path.join(self.photo_dir, photo_name)
        photo_smooth_path = os.path.join(self.photo_smooth_dir, photo_name)
        
        anime_path = os.path.join(self.anime_dir, anime_name)
        smooth_path = os.path.join(self.smooth_dir, anime_name)

        # 3. Đọc ảnh
        photo_img = self._read_img_pil(photo_path)
        photo_smooth_img = self._read_img_pil(photo_smooth_path)
        
        anime_img = self._read_img_pil(anime_path)
        smooth_img = self._read_img_pil(smooth_path)

        # 4. Transform đồng bộ cho từng cặp
        photo_img, photo_smooth_img = self.paired_transform(photo_img, photo_smooth_img)
        anime_img, smooth_img = self.paired_transform(anime_img, smooth_img)

        return photo_img, anime_img, smooth_img, photo_smooth_img