import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm


def make_edge_smooth(img_path, save_path):
    img = cv2.imread(img_path)
    if img is None:
        return

    # 1. Edge preservation smoothing (approximate LO smoothing)
    # sigma_s: Range of neighborhood (0-200), sigma_r: Range of colors (0-1)
    # We use edgePreservingFilter which is faster and effective for "cartoon" look prep
    smooth = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)

    # 2. Refine with details (Optional customized logic matching typical AnimeGAN preprocessing)
    # Can add dialation if needed, but edgeOscillating is usually enough for the 'smooth' target

    cv2.imwrite(save_path, smooth)


def main():
    parser = argparse.ArgumentParser(
        description='Edge Smoothing for AnimeGAN dataset')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to input anime images')
    parser.add_argument('--output_dir', type=str,
                        required=True, help='Path to save smooth images')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    files = os.listdir(args.input_dir)
    print(f"Processing {len(files)} images...")

    for f in tqdm(files):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        in_path = os.path.join(args.input_dir, f)
        out_path = os.path.join(args.output_dir, f)
        make_edge_smooth(in_path, out_path)

    print("Done.")


if __name__ == '__main__':
    main()
