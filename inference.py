import torch
import cv2
import argparse
import os
import numpy as np
from models.generator import Generator


def main():
    parser = argparse.ArgumentParser(description="AnimeGANv3 Inference")
    parser.add_argument('--checkpoint', type=str,
                        required=True, help="Path to checkpoint file")
    parser.add_argument('--input', type=str, required=True,
                        help="Path to input image or directory")
    parser.add_argument(
        '--output', type=str, default="results/", help="Path to output directory")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using NVIDIA GPU: {device}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print(f"No GPU found. Using CPU")

    G = Generator(in_channels=3).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    # the loaded state dict could be 'G' or just the raw model weights depending on format
    if 'G' in ckpt:
        G.load_state_dict(ckpt['G'])
    else:
        G.load_state_dict(ckpt)

    G.eval()
    os.makedirs(args.output, exist_ok=True)

    def process(img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Preprocessing aligned with TF logic (resize to nearest 8)
        h, w = img.shape[:2]

        def to_8s(x):
            return 256 if x < 256 else x - x % 8
        img = cv2.resize(img, (to_8s(w), to_8s(h)))
        img = img.astype(np.float32) / 127.5 - 1.0

        with torch.no_grad():
            img_t = torch.from_numpy(img).permute(
                2, 0, 1).unsqueeze(0).to(device)
            # Only generate the main tail representation directly, save inference parameters
            fake_m = G(img_t, inference=True)

        # Denormalize mapping
        out = (fake_m.squeeze(0).permute(1, 2, 0) + 1.0) / 2.0 * 255.0
        out = out.clamp(0, 255).cpu().numpy().astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return out

    if os.path.isdir(args.input):
        for name in os.listdir(args.input):
            if name.lower().endswith(('.jpg', '.png', '.jpeg')):
                out_img = process(os.path.join(args.input, name))
                if out_img is not None:
                    cv2.imwrite(os.path.join(args.output, name), out_img)
    else:
        out_img = process(args.input)
        if out_img is not None:
            cv2.imwrite(os.path.join(
                args.output, os.path.basename(args.input)), out_img)


if __name__ == '__main__':
    main()
