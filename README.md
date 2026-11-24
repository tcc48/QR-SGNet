# QR-SGNet

**A Structure-Guided Multi-Scale Network for Unreadable QR Code Restoration**

QR-SGNet is a structure-guided image restoration network designed to recover QR codes that standard decoders cannot read. Instead of optimizing pixel reconstruction alone, the network explicitly learns QR-code structures—module boundaries and binary module layouts—and feeds these structural predictions back into the decoder through a spatially adaptive feature gate.

This repository contains the PyTorch implementation of the model and its supervised training pipeline.

## Highlights

- **Structure-conditioned feature modulation.** Edge and binary-module predictions are converted into spatial guidance and fed back to the decoder features through a bounded gate.
- **Self-derived structural supervision.** Edge targets are computed from reference images with normalized Sobel gradients; binary-module targets are obtained by thresholding the reference images. No manual structural annotations are required.
- **Multi-scale restoration.** A three-scale encoder–decoder with channel-attention blocks reconstructs the input at its original spatial resolution.
- **Joint optimization.** The model is trained with reconstruction, auxiliary reconstruction, edge, and binary-structure losses.
- **Decoder-oriented evaluation.** The manuscript evaluates restored images using both OpenCV's `QRCodeDetector` and the WeChat QR-code detector, in addition to PSNR, SSIM, and inference-time measurements.


## Dataset

The experiments in the accompanying manuscript use the public QR-code dataset released with the benchmark study by Sancar. The dataset contains three groups:

| Dataset split | Number of images | Description | Use in the manuscript |
|---|---:|---|---|
| Paired learning set | 15,999 pairs | Degraded inputs paired with high-quality reference images | Supervised training and reconstruction assessment |
| Unreadable real scans | 4,593 | Printed and scanned QR codes that OpenCV could not decode before restoration | Decoder evaluation |
| Unreadable simulated images | 2,899 | Synthetically degraded QR codes that neither OpenCV nor WeChat could decode before restoration | Decoder evaluation |

**Dataset source:** [Figshare dataset (DOI: 10.6084/m9.figshare.28424213.v1)](https://doi.org/10.6084/m9.figshare.28424213.v1)

The training code expects paired low-quality and high-quality images in separate directories. It first matches images by relative path and then falls back to a unique filename stem. Unmatched images are not used. Each matched pair must have the same dimensions.

Expected layout (the directory names can be changed through command-line arguments):

```text
QR-SGNet/
├── data/
│   └── raw/
│       └── Learning Data/
│           ├── low/       # degraded QR-code images
│           └── high/      # corresponding reference images
├── v4/
│   ├── __init__.py
│   ├── data.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   └── train.py
└── README.md
```

The real-scan and simulated evaluation sets are used for decoder evaluation in the manuscript. The training script included here consumes the paired learning set only.

## Installation

The included code uses Python, PyTorch, NumPy, Pillow, and scikit-image. Exact dependency versions were not recorded in the supplied code archive; install a PyTorch build appropriate for your operating system and CUDA environment.

```bash
# Create and activate a virtual environment first, if desired.
python -m pip install numpy pillow scikit-image

# Install PyTorch using the command recommended by the official selector:
# https://pytorch.org/get-started/locally/
```

### Important: package directory name

The supplied archive places the Python files in a directory named `code/`, while `train.py` imports modules from `v4` (for example, `from v4.model import QRMSRNetV4`). To run the code without changing those imports, rename `code/` to `v4/` after extracting the archive.

Linux/macOS:

```bash
mv code v4
```

Windows PowerShell:

```powershell
Rename-Item .\code v4
```

The repository layout shown above reflects the directory name expected by the current training script.

## Training

From the repository root, run:

```bash
python -m v4.train \
  --low_dir "data/raw/Learning Data/low" \
  --high_dir "data/raw/Learning Data/high" \
  --epochs 150 \
  --batch_size 8 \
  --lr 2e-4 \
  --base 48 \
  --seed 42 \
  --val_ratio 0.2 \
  --workers 4 \
  --output outputs/qr_v4
```

On Windows PowerShell, the same command can be entered on one line, or use PowerShell's backtick for line continuation.

### Training options

| Argument | Default | Description |
|---|---:|---|
| `--low_dir` | `data/raw/Learning Data/low` | Directory of degraded input images |
| `--high_dir` | `data/raw/Learning Data/high` | Directory of reference images |
| `--epochs` | `150` | Maximum number of training epochs |
| `--batch_size` | `8` | Training and validation batch size |
| `--lr` | `2e-4` | AdamW learning rate |
| `--base` | `48` | Base feature width of the network |
| `--seed` | `42` | Random seed |
| `--val_ratio` | `0.2` | Fraction of pairs assigned to validation |
| `--workers` | `4` | DataLoader worker count |
| `--output` | `outputs/qr_v4` | Output directory for checkpoints and training history |
| `--edge_weight` | `0.10` | Weight of the edge loss |
| `--binary_weight` | `0.05` | Weight of the binary-structure loss |

For the 15,999-pair dataset and the default 20% validation ratio, the split is 12,799 training pairs and 3,200 validation pairs. The training dataset applies paired random horizontal flips, vertical flips, and rotations by multiples of 90 degrees.

On Windows, or when multiprocessing causes issues, try `--workers 0`.

### Optimization settings

The supplied training script uses:

- Optimizer: AdamW, weight decay \(1\times10^{-4}\).
- Initial learning rate: \(2\times10^{-4}\).
- Batch size: 8.
- Maximum epochs: 150.
- Gradient-norm clipping: 1.0.
- Scheduler: `ReduceLROnPlateau` on validation PSNR, with factor 0.5 and patience 6.
- Checkpoint selection: highest validation PSNR.
- Random seed: 42.



## Outputs

After training, the output directory contains:

```text
outputs/qr_v4/
├── checkpoints/
│   ├── best.pth    # checkpoint with the best validation PSNR
│   └── last.pth    # checkpoint from the final epoch
└── history.json   # per-epoch training/validation statistics
```

Each checkpoint stores the model state, training arguments, epoch, and best validation PSNR.

## Reported results

The following results are reported in the accompanying manuscript for QR-SGNet:


## License

No license file was included in the supplied code archive. Add a `LICENSE` file before redistributing this repository, and make sure the dataset terms are followed separately.
