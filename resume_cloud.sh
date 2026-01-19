#!/bin/bash
echo "Installing Dependencies (forcing overwrite of system packages)..."
pip install fastapi uvicorn python-multipart imageio opencv-python-headless omegaconf timm einops scikit-image trimesh pandas open3d joblib scipy matplotlib matplotlib-inline --ignore-installed

echo "Killing Jupyter (to free port 8888)..."
pkill -f jupyter

echo "Starting Server..."
cd /workspace/foundation-stereo-testing/
uvicorn cloud_server:app --host 0.0.0.0 --port 8888
