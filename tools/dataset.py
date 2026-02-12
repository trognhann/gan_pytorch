import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class AnimeDataset(Dataset):
    def __init__(self, root_dir, transform=None, img_size=256):
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size

        # Directories
        self.photo_dir = os.path.join(root_dir, 'train_photo')
        self.anime_dir = os.path.join(root_dir, 'anime_style')
        self.smooth_dir = os.path.join(root_dir, 'anime_smooth')

        # File lists
        self.photos = self._load_files(self.photo_dir)
        self.animes = self._load_files(self.anime_dir)
        self.smooths = self._load_files(self.smooth_dir)

        self.len_photo = len(self.photos)
        self.len_anime = len(self.animes)
        self.len_smooth = len(self.smooths)

        # Default Transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),  # Or RandomCrop
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def _load_files(self, dir_path):
        if not os.path.exists(dir_path):
            return []
        files = [f for f in os.listdir(dir_path) if f.lower().endswith(
            ('.jpg', '.jpeg', '.png', '.bmp'))]
        return files

    def __len__(self):
        return max(self.len_photo, self.len_anime)

    def __getitem__(self, idx):
        # Unpaired loading
        photo_idx = idx % self.len_photo
        anime_idx = idx % self.len_anime
        smooth_idx = idx % self.len_smooth

        photo_path = os.path.join(self.photo_dir, self.photos[photo_idx])
        anime_path = os.path.join(self.anime_dir, self.animes[anime_idx])
        smooth_path = os.path.join(self.smooth_dir, self.smooths[smooth_idx])

        photo_img = self._read_img(photo_path)
        anime_img = self._read_img(anime_path)
        smooth_img = self._read_img(smooth_path)

        if self.transform:
            photo_img = self.transform(photo_img)
            anime_img = self.transform(anime_img)
            smooth_img = self.transform(smooth_img)

        return photo_img, anime_img, smooth_img

    def _read_img(self, path):
        img = cv2.imread(path)
        if img is None:
            # Return a black image if failed (robustness)
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
