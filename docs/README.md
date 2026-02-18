# VICET (Virtual Contrast Enhancement Transformer)

## Introduction
VICET is a deep learning project that utilizes a DINOv3-based Transformer model to virtually generate contrast-enhanced effects similar to Contrast-Enhanced CT (CECT) from Non-Contrast CT (NCCT) images.
This model synthesizes visual effects as if a contrast agent was administered, based on high-dimensional features extracted from input images, and can be used for diagnostic assistance purposes.

## Key Features
- **Feature Extraction**: Uses DINOv3 (Vision Transformer) to extract strong semantic features from NCCT images.
- **Diff Decoder**: Predicts the difference (Diff Map) between the original image and the contrast-enhanced image from the extracted features.
- **Location Decoder**: Predicts the anatomical location (Location Map) where contrast enhancement will occur, suppressing noise in unnecessary areas.
- **Image Synthesis**: Combines the predicted Diff Map and Location Map and applies post-processing to generate the final virtual CECT image.

## Directory Structure
```
VICET/
├── data/
│   ├── dicom/            # Stores original DICOM files (NCCT, CECT)
│   ├── feature/          # Feature files extracted by DINOv3 (.npy)
│   ├── diff/             # Ground Truth Difference Maps (.npy)
│   └── location/         # Ground Truth Location Maps (.npy)
├── decoder_diff/         # Diff Decoder model and training code
│   ├── model.py          # Model architecture definition
│   ├── train.py          # Training script
│   └── checkpoints/      # Saved weights
├── decoder_location/     # Location Decoder model and training code
│   ├── model.py          # Model architecture definition
│   ├── train.py          # Training script
│   └── checkpoints/      # Saved weights
├── dinov3/               # DINOv3 model related libraries
├── dicom_to_feature.py   # Data preprocessing and feature extraction script
├── test_decoder.py       # Inference and result visualization script
├── train.sh              # Full model training execution script
└── verify.py             # Data verification script
```

## Installation
This project runs on Python 3.8+ and PyTorch environment. It is recommended to set up a virtual environment using Conda.

```bash
# Create and activate Conda environment
conda env create -f environment.yml
conda activate vicet
```

`tmux` must be installed to run the training script (`train.sh`). (Linux environment recommended)
```bash
sudo apt-get install tmux
```

## Usage

### 1. Data Preparation
Process DICOM files to generate DINOv3 features and Ground Truth maps required for training.
Before running `dicom_to_feature.py`, you must uncomment the `generate_feature` function call within the `main` function to generate input features.

```bash
# Run dicom_to_feature.py
python dicom_to_feature.py --dicom_dir ./data/dicom --feature_dir ./data/feature
```
- `--dicom_dir`: Path to the folder containing patient DICOM files
- `--feature_dir`: Path to save extracted features

### 2. Training
Train the Diff Decoder and Location Decoder in parallel using the `train.sh` script. This script creates a `tmux` session to train both models simultaneously.

**Note**: Check if the path inside `train.sh` (`/workspace/ViTN2C/...`) matches your current project path and modify it if necessary.

```bash
chmod +x train.sh
./train.sh
```

### 3. Inference & Visualization
Perform inference on a single case using the trained model and visualize the results.
You need to modify the file path and checkpoint path in the `if __name__ == "__main__":` block at the bottom of `test_decoder.py` to match your environment.

```bash
python test_decoder.py
```
The execution results will be saved in the `decoder_test_results/` directory as `all_outputs.png` image and result `.npy` files.

## Model Architecture
- **Encoder**: Embeds images using DINOv3 (ViT-L/16, etc.).
- **RefinedDiffDecoder**: Restores 1024-channel features to a 512x512 Diff Map through ResNet-style ConvBlocks and Upsampling layers.
- **RefinedLocationDecoder**: Has a structure similar to DiffDecoder and predicts the contrast enhancement area by binarizing it.

## License
MIT License
