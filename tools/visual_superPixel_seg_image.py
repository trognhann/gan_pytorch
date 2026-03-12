import os
import cv2
import argparse
import numpy as np
from skimage import segmentation, color
from tqdm import tqdm

def process_felzenszwalb(img_path, scale=5, sigma=0.8, min_size=50):
    img = cv2.imread(img_path)
    if img is None:
        return None
        
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_seg = segmentation.felzenszwalb(rgb, scale=scale, sigma=sigma, min_size=min_size)
    
    out_rgb = color.label2rgb(img_seg, rgb, bg_label=-1, kind='avg')
    
    out_bgr = cv2.cvtColor(out_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return out_bgr

def process_slic(img_path, seg_num=200):
    img = cv2.imread(img_path)
    if img is None:
        return None
        
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    seg_label = segmentation.slic(rgb, n_segments=seg_num, sigma=1, start_label=0,
                                  compactness=10, convert2lab=True)
    out_rgb = color.label2rgb(seg_label, rgb, bg_label=-1, kind='avg')
    
    out_bgr = cv2.cvtColor(out_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return out_bgr

def main():
    parser = argparse.ArgumentParser(description='Tạo ảnh SuperPixel (Region-smoothed) cho DTGAN')
    parser.add_argument('--input_dir', type=str, default=r'dataset\train_photo',
                        help='Đường dẫn thư mục ảnh thực tế (vd: dataset/train_photo)')
    parser.add_argument('--output_dir', type=str, default=r'dataset\train_photo_superpixel', 
                        help='Đường dẫn lưu kết quả (vd: dataset/photo_superpixel)')
    parser.add_argument('--method', type=str, default='felzenszwalb', choices=['felzenszwalb', 'slic'],
                        help='Thuật toán phân mảnh (mặc định: felzenszwalb)')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Đang xử lý {len(files)} ảnh bằng thuật toán {args.method.upper()}...")

    for f in tqdm(files):
        in_path = os.path.join(args.input_dir, f)
        out_path = os.path.join(args.output_dir, f)
        
        if args.method == 'felzenszwalb':
            result_img = process_felzenszwalb(in_path, scale=5, sigma=0.8, min_size=50)
        else:
            result_img = process_slic(in_path, seg_num=200)
            
        if result_img is not None:
            cv2.imwrite(out_path, result_img)

    print("Hoàn tất tạo tập dữ liệu Region-smoothed!")

if __name__ == '__main__':
    main()