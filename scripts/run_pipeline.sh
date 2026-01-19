#!/bin/bash
set -e

# 1. Capture
echo "📸 Capturing new images..."
../.venv/bin/python capture_stereo.py

# 2. Inference
echo "🧠 Running FoundationStereo (CPU Mode)..."
CUDA_VISIBLE_DEVICES="" ../.venv/bin/python run_demo.py \
    --ckpt_dir ../pretrained_models/23-51-11/11-33-40/model_best_bp2.pth \
    --scale 0.5 \
    --left_file ../assets/my_left.png \
    --right_file ../assets/my_right.png \
    --intrinsic_file ../assets/my_K.txt

# 3. Open
echo "🌐 Opening viewer..."
xdg-open "http://localhost:8000/view_cloud.html"
