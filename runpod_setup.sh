#!/bin/bash
set -e

echo "Starting FoundationStereo setup on RunPod..."

# 1. Install system dependencies for OpenCV
echo "Installing system dependencies..."
apt-get update
apt-get install -y libgl1-mesa-glx libglib2.0-0 git zip unzip

# 2. Install Python dependencies
echo "Installing Python requirements..."
# Install gdown first to download weights
pip install gdown
# Install project requirements
pip install -r requirements.txt
# Streaming & Fixes
pip install --ignore-installed blinker
pip install imageio opencv-python-headless scikit-image open3d timm einops omegaconf trimesh pandas joblib fastapi uvicorn python-multipart

# 3. Download Model Weights
echo "Downloading Model Weights (23-51-11)..."
mkdir -p pretrained_models
cd pretrained_models
# Download the specific folder '23-51-11' from the provided Google Drive link
gdown --folder "https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing"
# The download creates a '23-51-11' folder. Just ensure structure matches what run_demo.py expects.
# Expected: ./pretrained_models/23-51-11/model_best_bp2.pth
# Move things if necessary.
if [ -d "FoundationStereo_checkpoint/23-51-11" ]; then
    # Sometimes gdown names it differently, checking structure.
    mv FoundationStereo_checkpoint/23-51-11 .
    rm -rf FoundationStereo_checkpoint
fi
cd ..

echo "Setup Complete! You can now run the demo."
