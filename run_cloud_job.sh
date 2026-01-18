#!/bin/bash
set -e

# 0. Safety measure: ensure deps are installed (fixes imageio error)
# 0. Safety measure: ensure deps are installed
# Fix for "Cannot uninstall blinker" error
pip install --ignore-installed blinker
pip install imageio opencv-python-headless scikit-image open3d timm einops omegaconf

# 1. Check for assets
if [ ! -f "assets.zip" ]; then
    echo "Error: assets.zip not found! Please upload it first."
    exit 1
fi

# 2. Unzip assets
echo "Unzipping assets..."
unzip -o assets.zip -d assets_cloud/

# 2.5 Handle potential 'assets' subfolder in the zip
if [ -d "assets_cloud/assets" ]; then
    mv assets_cloud/assets/* assets_cloud/
    rmdir assets_cloud/assets
fi

# 3. Run Inference
echo "Running FoundationStereo (Standard Mode)..."
python scripts/run_demo.py \
  --left_file assets_cloud/my_left.png \
  --right_file assets_cloud/my_right.png \
  --intrinsic_file assets_cloud/my_K.txt \
  --ckpt_dir pretrained_models/23-51-11/model_best_bp2.pth \
  --out_dir output_cloud \
  --scale 1.0

# 4. Pack Results
echo "Zipping results..."
zip -j results.zip output_cloud/vis.png output_cloud/cloud.ply

echo "------------------------------------------------"
echo "DONE! Right-click 'results.zip' and Download."
echo "------------------------------------------------"
