# Railway Track Fault Detection using Vision Transformer

**EE655: Computer Vision — Course Project, IIT Kanpur**

## Authors
- **Akash Tiwari** (Roll No: 241090403)
- **Vaibhav Soni** (Roll No: 241090419)

## Overview
A custom Vision Transformer (ViT) implemented from scratch in PyTorch for binary classification of railway track images into **Defective** and **Non-defective** categories. The file `railway_vit.py` applies pure ViT architecture on the dataset whereas the file `efficientnet_vit.py` applies the weights from trained EfficientNet-B0 model on the dataset for feature extraction then using the weights for the embedding in the vision transformer.

## Architecture
| Component | Details |
|-----------|---------|
| Input Size | 224 × 224 × 3 (RGB) |
| Patch Size | 16 × 16 |
| Number of Patches | 196 |
| Embedding Dimension | 128 |
| Attention Heads | 4 |
| Transformer Blocks | 6 |
| MLP Hidden Dimension | 256 |
| Total Parameters | 919,170 |

## Dataset
Kaggle Identifier: https://doi.org/10.34740/kaggle/dsv/1884733

Railway Track Fault Detection dataset with the following split:

| Split | Defective | Non-defective | Total |
|-------|-----------|---------------|-------|
| Train | 150 | 150 | 300 |
| Validation | 31 | 31 | 62 |
| Test | 11 | 11 | 22 |

## Results
- FROM VIT
- **Best Validation Accuracy:** 80.65% (Epoch 16)
- **Training Accuracy:** ~82% (Epoch 30)

## Data Augmentation
- Random Horizontal Flip
- Random Rotation (±15°)
- Color Jitter (brightness, contrast, saturation ±0.2)
- Random Affine Translation (±10%)
- ImageNet Normalization

## Requirements
```
torch
torchvision
scikit-learn
numpy
```

## Usage
```bash
# Place the dataset in archive/ directory
run python railway_vit.py 
```

## Project Structure
```
├── railway_vit.py          # Main training script (ViT from scratch)
├── archive/                # Dataset directory
│   └── Railway Track fault Detection Updated/
│       ├── Train/
│       │   ├── Defective/
│       │   └── Non defective/
│       ├── Validation/
│       │   ├── Defective/
│       │   └── Non defective/
│       └── Test/
│           ├── Defective/
│           └── Non defective/
├── writeup/                # CVPR-format LaTeX write-up
│   └── main.tex
└── README.md
```

## License
MIT
