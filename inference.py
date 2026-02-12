import argparse
import os
import torch
import cv2
import numpy as np
from net.generator import Generator
from tools.utils import denormalize
from torchvision import transforms
from tqdm import tqdm


def process_image(img, model, device):
    # Preprocess
    h, w = img.shape[:2]
    # Resize to multiple of 32 for UNet/Generator
    new_h = (h // 32) * 32
    new_w = (w // 32) * 32
    img_resized = cv2.resize(img, (new_w, new_h))

    img_t = transforms.ToTensor()(img_resized)
    img_t = transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img_t)
    img_t = img_t.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_t, training=False)

    output = denormalize(output.cpu()[0])
    output = output.permute(1, 2, 0).numpy() * 255
    output = output.astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    # Resize back to original? Or keep? Usually keep new size or resize back.
    # Let's resize back to original h, w
    output = cv2.resize(output, (w, h))
    return output


def main():
    parser = argparse.ArgumentParser(description='AnimeGANv3 Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to generator checkpoint')
    parser.add_argument('--source', type=str, required=True,
                        help='Input image or directory')
    parser.add_argument('--dest', type=str, required=True,
                        help='Output directory')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Model
    netG = Generator().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    netG.load_state_dict(checkpoint['model_state_dict'])
    netG.eval()

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    if os.path.isfile(args.source):
        files = [args.source]
        root = os.path.dirname(args.source)
    else:
        files = [os.path.join(args.source, f) for f in os.listdir(
            args.source) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        root = args.source

    print(f"Processing {len(files)} images...")

    for f in tqdm(files):
        img = cv2.imread(f)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        anime_img = process_image(img, netG, device)

        save_name = os.path.basename(f)
        cv2.imwrite(os.path.join(args.dest, save_name), anime_img)

    print("Done.")


if __name__ == '__main__':
    main()
