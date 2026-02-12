# Double-Tail AnimeGANv3 (PyTorch)

A PyTorch implementation of Double-Tail Generative Adversarial Network (DTGAN) / AnimeGANv3 for high-quality photo-to-anime conversion, optimized for human faces.

## 1. Project Structure
```
AnimeGANv3/
├── net/                # Model architecture
│   ├── common.py       # Common blocks (LADE, Conv, etc.)
│   ├── generator.py    # Double-tail generator
│   └── discriminator.py # PatchGAN discriminators
├── tools/              # Tools for data and processing
│   ├── utils.py        # Utilities (IO, checkpoints, logging)
│   └── edge_smooth.py  # Edge smoothing for training data
├── utils/              # Helper functions
├── train.py            # Main training script
├── inference.py        # Inference script
└── README.md           # This file
```

## 2. Environment Setup
Prerequisites:
- Python 3.7+
- PyTorch 1.7+
- Torchvision
- OpenCV
- NumPy
- tqdm
- argparse

```bash
pip install torch torchvision opencv-python numpy tqdm
```

## 3. Data Preparation
Structure your dataset as follows:
```
dataset/
├── train_photo/        # Real photos (faces)
├── anime_style/        # Anime style images
├── anime_smooth/       # Smooth version of anime images (generated via edge_smooth.py)
└── test/               # Test photos
```

**Step 3.1: Generate Smooth Images**
Run the edge smoothing script to create the `anime_smooth` dataset (helps remove noise in style images).
```bash
python tools/edge_smooth.py --input_dir dataset/anime_style --output_dir dataset/anime_smooth
```

## 4. Training
Train the model using `train.py`. The training process includes:
1. **Pre-training**: Initialize the generator with content loss only (optional but recommended for stability).
2. **GAN Training**: Train both Generator and Discriminator.

```bash
python train.py --dataset dataset --batch_size 8 --epochs 100 --checkpoint_dir checkpoints
```

**Arguments:**
- `--dataset`: Path to dataset root.
- `--batch_size`: Batch size (default: 8).
- `--epochs`: Number of epochs (default: 100).
- `--lr`: Learning rate (default: 2e-4).
- `--resume`: Resume from latest checkpoint (default: False).
- `--checkpoint_dir`: Directory to save models.
- `--log_dir`: Directory to save logs.

## 5. Inference
Run inference on a single image or a folder of images.

```bash
python inference.py --checkpoint checkpoints/best_netG.pth --source test_image.jpg --dest output.jpg
```

**Arguments:**
- `--checkpoint`: Path to trained generator weights.
- `--source`: Input image path or directory.
- `--dest`: Output path or directory.

## 6. Model Architecture
- **Generator**: Double-tail architecture (Support Tail & Main Tail) with External Attention.
- **Discriminator**: Patch-level discriminators.
- **Normalization**: Linearly Adaptive Denormalization (LADE).
