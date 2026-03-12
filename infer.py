"""
AnimeGANv3 PyTorch — Inference Script
======================================
Convert real photos to anime style using a trained Generator.

Usage:
  # Single image
  python infer.py --checkpoint checkpoint/latest_G.pth --input photo.jpg --output result.jpg

  # Batch (folder)
  python infer.py --checkpoint checkpoint/latest_G.pth --input photos/ --output results/

  # Custom resolution
  python infer.py --checkpoint checkpoint/latest_G.pth --input photo.jpg --output result.jpg --img_size 512
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

from models.generator import Generator


def load_generator(checkpoint_path, device):
    """Load Generator from checkpoint (only model weights needed)."""
    G = Generator(in_channels=3).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if saved with DataParallel
    new_state = {}
    for k, v in state_dict.items():
        new_state[k.replace('module.', '')] = v

    G.load_state_dict(new_state, strict=False)
    G.eval()
    return G


def preprocess(img_path, img_size=None):
    """Read image, optionally resize, normalize to [-1, 1]."""
    img = Image.open(img_path).convert('RGB')
    original_size = img.size  # (W, H)

    if img_size is not None:
        img = TF.resize(img, (img_size, img_size))

    tensor = TF.to_tensor(img)  # [0, 1]
    tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
    return tensor.unsqueeze(0), original_size


def postprocess(tensor, original_size=None):
    """Denormalize [-1,1] tensor → PIL Image, optionally resize back."""
    img = tensor.squeeze(0).cpu().clamp(-1, 1)
    img = (img + 1.0) / 2.0  # [0, 1]
    img = TF.to_pil_image(img)
    if original_size is not None:
        img = img.resize(original_size, Image.LANCZOS)
    return img


@torch.no_grad()
def infer_single(G, img_path, output_path, img_size, device, keep_size=True):
    """Run inference on a single image."""
    tensor, original_size = preprocess(img_path, img_size)
    tensor = tensor.to(device)

    fake_m = G(tensor, inference=True)

    result = postprocess(fake_m, original_size if keep_size else None)
    result.save(output_path, quality=95)
    return result


def main():
    parser = argparse.ArgumentParser(description='AnimeGANv3 PyTorch Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to Generator checkpoint (.pth)')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image path or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output image path or directory')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Resize input to this size for inference (default: 256)')
    parser.add_argument('--keep_size', action='store_true', default=True,
                        help='Resize output back to original input size')
    parser.add_argument('--no_keep_size', dest='keep_size', action='store_false',
                        help='Keep output at inference resolution')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    G = load_generator(args.checkpoint, device)
    print("Generator loaded successfully.")

    # Single image or batch
    if os.path.isfile(args.input):
        # Single image
        print(f"Processing: {args.input}")
        infer_single(G, args.input, args.output, args.img_size, device, args.keep_size)
        print(f"Saved: {args.output}")

    elif os.path.isdir(args.input):
        # Batch
        os.makedirs(args.output, exist_ok=True)
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        files = [f for f in sorted(os.listdir(args.input))
                 if f.lower().endswith(exts)]

        print(f"Processing {len(files)} images from {args.input}")
        for fname in tqdm(files):
            in_path = os.path.join(args.input, fname)
            out_path = os.path.join(args.output, fname)
            infer_single(G, in_path, out_path, args.img_size, device, args.keep_size)

        print(f"Done! Results saved to {args.output}")
    else:
        print(f"Error: {args.input} is not a valid file or directory.")


if __name__ == '__main__':
    main()
