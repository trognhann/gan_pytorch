import os
import cv2
import argparse
from tqdm import tqdm

def resize_dataset(input_dir, output_dir, target_size=256):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    print(f"Đang xử lý {len(files)} ảnh từ thư mục: {input_dir}")

    for f in tqdm(files):
        in_path = os.path.join(input_dir, f)
        out_path = os.path.join(output_dir, f)
        
        img = cv2.imread(in_path)
        if img is None:
            continue
            
        resized_img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        cv2.imwrite(out_path, resized_img)

def main():
    parser = argparse.ArgumentParser(description='Resize Dataset for DTGAN (AnimeGANv3) Training')
    parser.add_argument('--input_dir', type=str,
                        default=r'dataset\train_photo',
                        help='Đường dẫn thư mục chứa ảnh gốc (vd: dataset/ghbli_c1_512)')
    parser.add_argument('--output_dir', type=str,
                        default=r'dataset\train_photo_256',
                        help='Đường dẫn thư mục lưu ảnh đã resize (vd: dataset/ghbli_c1_256)')
    parser.add_argument('--size', type=int, default=256, 
                        help='Kích thước ảnh mục tiêu (mặc định: 256)')
    args = parser.parse_args()

    resize_dataset(args.input_dir, args.output_dir, args.size)
    print("Hoàn tất! Dataset đã sẵn sàng cho DTGAN.")

if __name__ == '__main__':
    main()