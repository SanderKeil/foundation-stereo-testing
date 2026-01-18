
import os
import io
import cv2
import torch
import numpy as np
import imageio
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from contextlib import asynccontextmanager
import argparse
from omegaconf import OmegaConf

# --- Import FoundationStereo Modules ---
import sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
from core.foundation_stereo import FoundationStereo
from Utils import vis_disparity

# --- Global Model State ---
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model
    print("Loading FoundationStereo Model...")
    
    ckpt_path = "pretrained_models/23-51-11/model_best_bp2.pth"
    # Fallback search if not found directly
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}, searching...")
        found = False
        for root, dirs, files in os.walk("pretrained_models"):
            if "model_best_bp2.pth" in files:
                ckpt_path = os.path.join(root, "model_best_bp2.pth")
                print(f"Found checkpoint at: {ckpt_path}")
                found = True
                break
        if not found:
            raise FileNotFoundError("Could not find model_best_bp2.pth inside pretrained_models/")
    # Try different config locations
    cfg_locations = [
        "pretrained_models/23-51-11/cfg.yaml",
        "model_configuration.yaml",
        "cfg.yaml"
    ]
    
    cfg = None
    for loc in cfg_locations:
        if os.path.exists(loc):
            print(f"Found config at: {loc}")
            cfg = OmegaConf.load(loc)
            break
            
    if cfg is None:
        raise FileNotFoundError("Could not find cfg.yaml or model_configuration.yaml!")

    # Merge with runtime args if needed, or just convert to Namespace
    # FoundationStereo expects 'args' to be namespace-like or dict-like that allows dot access
    # OmegaConf object allows dot access.
    
    # We need to add runtime args that aren't in the yaml
    cfg.mixed_precision = True
    cfg.restore_ckpt = ckpt_path
    
    # Missing args from error log:
    if not hasattr(cfg, 'vit_size'):
        cfg.vit_size = 'vitl' # Changed from 'base' to 'vitl' based on extractor.py
    if not hasattr(cfg, 'n_downsample'):
        cfg.n_downsample = 2
    if not hasattr(cfg, 'n_gru_layers'):
        cfg.n_gru_layers = 3
    if not hasattr(cfg, 'corr_radius'):
        cfg.corr_radius = 4
    if not hasattr(cfg, 'corr_levels'):
        cfg.corr_levels = 4
    
    # Initialize model
    model = FoundationStereo(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(model, device_ids=[0])
    model.to(device)
    
    # Load weights
    print(f"Loading weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print("Model Loaded Successfully!")
    yield
    # Clean up
    del model

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "Model Ready", "device": "cuda" if torch.cuda.is_available() else "cpu"}

@app.post("/process")
async def process_stereo(left_image: UploadFile = File(...), right_image: UploadFile = File(...)):
    global model
    
    if model is None:
        return Response(content="Model not loaded", status_code=500)

    # Read Images
    left_bytes = await left_image.read()
    right_bytes = await right_image.read()
    
    img_left = imageio.imread(left_bytes)
    img_right = imageio.imread(right_bytes)
    
    # Prepare tensors
    img_left = torch.from_numpy(img_left).permute(2, 0, 1).float() / 255.0
    img_right = torch.from_numpy(img_right).permute(2, 0, 1).float() / 255.0
    
    # Batch dimension
    img_left = img_left.unsqueeze(0)
    img_right = img_right.unsqueeze(0)
    
    device = next(model.parameters()).device
    
    # FoundationStereo Forward
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            # Model forward
            # The model signature: forward(image1, image2, iters=12, ...)
            # We pass tensors directly
            output = model(img_left.to(device), img_right.to(device), test_mode=True)
            
            # Output in test_mode=True returns 'disp_up' tensor directly (see foundation_stereo.py line 258)
            disp = output
            
            # Post-processing
            disp = disp[0,0].cpu().numpy() # (H, W)
            
            # Colorize
            vis = vis_disparity(disp)
            
            # Encode response
            res, im_png = cv2.imencode(".jpg", vis)
            return Response(content=im_png.tobytes(), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
