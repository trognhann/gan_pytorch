import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm

def make_edge_smooth_with_noise(img_path, save_path, img_size=256, kernel_size=5, noise_prob=0.1):
    bgr_img = cv2.imread(img_path)
    gray_img = cv2.imread(img_path, 0)
    
    if bgr_img is None or gray_img is None:
        return

    bgr_img = cv2.resize(bgr_img, (img_size, img_size))
    gray_img = cv2.resize(gray_img, (img_size, img_size))

    edges = cv2.Canny(gray_img, 100, 200)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilation = cv2.dilate(edges, kernel)

    blurred_img = cv2.GaussianBlur(bgr_img, (kernel_size, kernel_size), 0)

    mask = np.expand_dims(dilation > 0, axis=-1)
    gauss_img = np.where(mask, blurred_img, bgr_img)

    noise_mask = np.random.rand(img_size, img_size)
    gauss_img[noise_mask < noise_prob] = 255 

    cv2.imwrite(save_path, gauss_img.astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description='Fast Edge Smooth + Noise for DTGAN dataset')
    parser.add_argument('--input_dir', type=str, default=r'dataset\style',
                        help='Đường dẫn thư mục ảnh anime gốc (vd: dataset/ghbli_c1)')
    parser.add_argument('--output_dir', type=str, default=r'dataset\style_smooth_noise', 
                        help='Đường dẫn lưu ảnh (vd: dataset/ghbli_smooth_noise)')
    parser.add_argument('--img_size', type=int, default=256, 
                        help='Kích thước ảnh mục tiêu (mặc định: 256)')
    parser.add_argument('--prob', type=float, default=0.1, 
                        help='Tỷ lệ nhiễu (mặc định: 0.1 - tương đương 10% diện tích)')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    files = os.listdir(args.input_dir)
    print(f"Đang xử lý {len(files)} ảnh để tạo tập Negative Samples...")

    for f in tqdm(files):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        in_path = os.path.join(args.input_dir, f)
        out_path = os.path.join(args.output_dir, f)
        
        make_edge_smooth_with_noise(in_path, out_path, img_size=args.img_size, noise_prob=args.prob)

    print("Hoàn tất! Thư mục smooth_noise đã sẵn sàng.")

if __name__ == '__main__':
    main()