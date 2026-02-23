import torch
import numpy as np
import random
import os
import torch.distributed as dist
from skimage import segmentation, color
import cv2
from l0_smoothing import L0Smoothing
import sys
import argparse
import yaml


def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_ddp():
    """Setup Distributed Data Parallel."""
    if 'WORLD_SIZE' in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return local_rank, rank, dist.get_world_size()
    else:
        return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def denormalize(x):
    """
    Convert [-1.0, 1.0] torch float tensor to [0, 255] numpy uint8 image
    Handling batch or single image (C, H, W) or (B, C, H, W).
    """
    x = (x + 1.0) / 2.0 * 255.0
    x = torch.clamp(x, 0, 255)

    if x.ndimension() == 4:
        # B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1)
    elif x.ndimension() == 3:
        # C, H, W -> H, W, C
        x = x.permute(1, 2, 0)

    return x.detach().cpu().numpy().astype(np.uint8)


def get_seg(batch_image):
    """
    Superpixel segmentation (Felzenszwalb)
    batch_image: numpy array [B, H, W, C] in range [-1.0, 1.0]
    """
    def get_superpixel(image):
        image = (image + 1.) * 127.5
        image = np.clip(image, 0, 255).astype(np.uint8)
        image_seg = segmentation.felzenszwalb(
            image, scale=5, sigma=0.8, min_size=50)
        image = color.label2rgb(
            image_seg, image, bg_label=-1, kind='avg').astype(np.float32)
        image = image / 127.5 - 1.0
        return image

    num_job = np.shape(batch_image)[0]
    # No joblib parallelization to avoid locking issues in PyTorch dataloaders,
    # Can be optimized or pre-processed offline.
    batch_out = [get_superpixel(img) for img in batch_image]
    return np.array(batch_out)


def get_NLMean_l0(batch_image):
    """
    NL-means + L0 smoothing.
    batch_image: numpy array [B, H, W, C] in range [-1.0, 1.0]
    """
    def process_revision(image):
        image = ((image + 1) * 127.5).clip(0, 255).astype(np.uint8)
        image = cv2.fastNlMeansDenoisingColored(image, None, 5, 6, 5, 7)
        image = L0Smoothing(image / 255.0, 0.005).astype(np.float32) * 2. - 1.
        return image.clip(-1., 1.)

    batch_out = [process_revision(img) for img in batch_image]
    return np.array(batch_out)
