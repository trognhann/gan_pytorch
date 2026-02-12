import torch
import os
import logging
import cv2
import numpy as np


def save_checkpoint(model, optimizer, epoch, step, checkpoint_dir, name='latest_net'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, f'{name}.pth')

    # Handle DataParallel
    if isinstance(model, torch.nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)


def load_checkpoint(model, optimizer, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return 0, 0

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint.get('epoch', 0), checkpoint.get('step', 0)


def get_logger(log_dir, name='train'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, f'{name}.log'))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def denormalize(x):
    # [-1, 1] -> [0, 1] -> [0, 255]
    out = (x + 1) / 2
    return out.clamp(0, 1)


def save_images(images, path, nrow=None):
    from torchvision.utils import save_image
    save_image(denormalize(images), path, nrow=nrow)
