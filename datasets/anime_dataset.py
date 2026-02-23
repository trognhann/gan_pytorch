import os
import cv2
import glob
import torch
import numpy as np
from torch.utils.data import Dataset


class AnimeDataset(Dataset):
    def __init__(self, dataset_dir, dataset_name, img_size=(256, 256)):
        super(AnimeDataset, self).__init__()
        self.img_size = img_size

        photo_dir = os.path.join(dataset_dir, "train_photo")
        style_dir = os.path.join(dataset_dir, dataset_name, "style")
        smooth_dir = os.path.join(dataset_dir, dataset_name, "smooth")
        seg_dir = os.path.join(dataset_dir, "seg_train_5-0.8-50")

        self.photo_paths = self._get_paths(photo_dir)
        self.style_paths = self._get_paths(style_dir)
        self.smooth_paths = self._get_paths(smooth_dir)
        self.seg_dir = seg_dir

        # Typically run for 1 epoch over the max dataset length
        self.num_images = max(len(self.photo_paths), len(self.style_paths))
        if self.num_images == 0:
            print("Warning: No images found. Check your dataset_dir in config.")

    def _get_paths(self, d):
        extensions = ('*.jpg', '*.jpeg', '*.png')
        paths = []
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(d, ext)))
            paths.extend(glob.glob(os.path.join(d, ext.upper())))
        return sorted(paths)

    def __len__(self):
        return self.num_images

    def preprocess(self, img, x8=True):
        if img is None:
            return np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.float32)
        h, w = img.shape[:2]
        if x8:
            def to_8s(x):
                return 256 if x < 256 else x - x % 8
            img = cv2.resize(img, (to_8s(w), to_8s(h)))
        else:
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        return img.astype(np.float32) / 127.5 - 1.0

    def __getitem__(self, idx):
        p_idx = np.random.randint(len(self.photo_paths))
        a_idx = np.random.randint(len(self.style_paths))

        photo_path = self.photo_paths[p_idx]
        style_path = self.style_paths[a_idx]

        # Smooth and style paths are assumed aligned by name in respective dirs
        # Try to find corresponding smooth file, otherwise pick randomly or fallback to pure style
        style_name = os.path.basename(style_path)
        smooth_path = os.path.join(os.path.dirname(
            self.smooth_paths[0]), style_name) if len(self.smooth_paths) > 0 else style_path
        if not os.path.exists(smooth_path) and len(self.smooth_paths) > 0:
            smooth_path = self.smooth_paths[np.random.randint(
                len(self.smooth_paths))]

        photo_name = os.path.basename(photo_path)
        seg_path = os.path.join(self.seg_dir, photo_name)

        p_img = cv2.imread(photo_path)
        # fallback to photo if seg missing
        seg_img = cv2.imread(seg_path) if os.path.exists(seg_path) else p_img
        s_img = cv2.imread(style_path)
        sm_img = cv2.imread(smooth_path)

        if p_img is not None:
            p_img = cv2.cvtColor(p_img, cv2.COLOR_BGR2RGB)
        if seg_img is not None:
            seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
        if s_img is not None:
            s_img = cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB)
        if sm_img is not None:
            sm_img = cv2.cvtColor(sm_img, cv2.COLOR_BGR2RGB)

        p_img = self.preprocess(p_img, x8=False)
        seg_img = self.preprocess(seg_img, x8=False)
        s_img = self.preprocess(s_img, x8=False)
        sm_img = self.preprocess(sm_img, x8=False)

        return {
            'photo': torch.from_numpy(p_img).permute(2, 0, 1),
            'photo_seg': torch.from_numpy(seg_img).permute(2, 0, 1),
            'style': torch.from_numpy(s_img).permute(2, 0, 1),
            'smooth': torch.from_numpy(sm_img).permute(2, 0, 1)
        }
