# AnimeGANv3 - PyTorch Implementation

A clean, modern, production-ready PyTorch (>= 2.1) reimplementation of AnimeGANv3. This repository was refactored strictly from the original TensorFlow 1.x codebase to reproduce identical algorithmic logic, model architectures, and custom numerical routines in PyTorch.

## Core Features
- **Accurate Replication**: Features the **Double-tail Generator** (`~1.02M` parameters), `ExternalAttention_v3`, and the PathGAN discriminator with Spectral Normalization.
- **Custom LADE Module**: Linear Adaptive Instance Denormalization matches analytical exact variance/mean extraction spatially.
- **Precision Mathematics**: Exact translations for LAB color space loss, VGG19 feature extraction, Grayscale Gram Matrices, and LSGAN calculations natively in exact PyTorch tensors.
- **Performance Focused**: Includes `DistributedDataParallel` (DDP) for multi-GPU training and native Mixed Precision (`torch.amp.GradScaler`). 
- **Efficient Inference**: The inference script optimizes generator processing strictly by dropping the auxiliary support-tail processing.
- **Cross-Platform**: Automatic device mapping. Will use **CUDA** on Linux/Windows NVIDIA cards, **MPS** on Apple Silicon Macs, and **CPU** if no accelerator is available.

---

## ⚙️ Installation

Make sure you have an environment with **Python 3.10+** and **PyTorch >= 2.1**. Run:

```bash
pip install torch torchvision torchaudio opencv-python numpy scikit-image pyyaml tensorboard tqdm
```

---

## 📂 Project Structure
```text
.
├── config.yaml          # Hyperparameters, batch size, epochs, paths, loss weights
├── train.py             # Main DDP training loop with Mixed Precision and TensorBoard log
├── inference.py         # Script to generate sample photos purely using the main generator tail
├── verify.py            # Simple sanity tests on network shapes and parameter sizes (~1.02M)
├── utils.py             # DDP bootstrap, set_seed, and auxiliary config mapping logic
├── l0_smoothing.py      # Computes the L0 smoothing for pseudo ground-truth images
├── models/
│   ├── generator.py     # Double-tail Generator definitions matching ReflectPad asymmetric constraints
│   ├── discriminator.py # Spectral Normalized PatchGAN 
│   ├── attention.py     # Implicit 1D Convolutions -> normalized Softmax Matrix Multiplication
│   ├── lade.py          # Spatial computation exact logic matching tf.instance_norm parameters
│   └── guided_filter.py # Differential Guided Filter mappings
├── losses/
│   ├── content.py       # VGG19 L1 Content definitions
│   ├── style.py         # Centered Grayscale Gram Matrix
│   ├── color_lab.py     # Matrix mapped analytical differentiable RGB to LAB comparisons
│   ├── region_smoothing.py # Superpixel comparisons mapped via VGG features
│   └── gan_loss.py      # Independent LSGAN formulations mapping Support / Main tail
└── datasets/
    └── anime_dataset.py # DDP-supported Dataset loading (handles 4-channel pseudo-GT dictionaries)
```

---

## 🏋️ Training (Train)

All settings (folder roots, learning rates, iteration numbers, loss ratios) are configured in `config.yaml`. Update the `dataset_dir` parameter located in `config.yaml` to point to the directory encompassing your `train_photo` and styled databases.

### Standard CPU / GPU (Automatic Inference Context)
```bash
python train.py --config config.yaml
```

### Multi-GPU (Distributed Data Parallel via `torchrun`)
To train using multiple NVIDIA GPUs, PyTorch handles parallel processes optimally:
```bash
torchrun --nproc_per_node=<num_gpus> train.py --config config.yaml
```

Training parameters, images, and model metrics will automatically route to `logs/` (readable via TensorBoard) and model weights periodically overwrite/save inside the `checkpoint/` directory as `.pt` states.

---

## 🎨 Inference (Test)

During deployment or inference, we strip away the Support Tail since it is only an auxiliary regularizing module to bound outputs for the Main Tail. `inference.py` ensures lightweight parameter invocation. 

Point the input flag to either a single photo or a directory.
```bash
python inference.py \
    --checkpoint checkpoint/AnimeGANv3_ep99.pt \
    --input ./dataset/val \
    --output ./results
```

---

## 📝 Configuration Options (`config.yaml`)

- `dataset`: Name of the Anime dataset collection (i.e `Hayao`, `Shinkai`).
- `init_epochs` & `epochs`: Regulates when the framework pivots from initialization (content-loss only matching via Guided Filters) towards adversarial updates.
- `checkpoint_dir` : Output parameter where the `.pt` models are saved across batches. You can manipulate this if testing on Kaggle or a cloud provider needing specific mount volumes (i.e `/kaggle/working`).
