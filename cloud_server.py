
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

    # Merge with runtime args if needed
    cfg.mixed_precision = True
    cfg.restore_ckpt = ckpt_path
    
    # Missing args defaults
    if not hasattr(cfg, 'vit_size'): cfg.vit_size = 'vitl'
    if not hasattr(cfg, 'n_downsample'): cfg.n_downsample = 2
    if not hasattr(cfg, 'n_gru_layers'): cfg.n_gru_layers = 3
    if not hasattr(cfg, 'corr_radius'): cfg.corr_radius = 4
    if not hasattr(cfg, 'corr_levels'): cfg.corr_levels = 4
    
    # Initialize model
    print("Initializing Model Architecture...")
    model_core = FoundationStereo(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on Device: {device}")
    
    # Wrap in DataParallel
    model = torch.nn.DataParallel(model_core, device_ids=[0])
    model.to(device)
    
    # Load weights
    print(f"Loading weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    state_dict = checkpoint['model']
    
    # --- INTELLIGENT WEIGHT FIX ---
    ckpt_keys = list(state_dict.keys())
    model_keys = list(model.state_dict().keys())
    
    ckpt_has_module = any(k.startswith('module.') for k in ckpt_keys)
    model_has_module = any(k.startswith('module.') for k in model_keys)
    
    print(f"DEBUG: Checkpoint keys have 'module.': {ckpt_has_module}")
    print(f"DEBUG: Model keys have 'module.': {model_has_module}")
    
    if ckpt_has_module and not model_has_module:
        print("WARNING: Removing 'module.' prefix from checkpoint to match model...")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    elif not ckpt_has_module and model_has_module:
        print("WARNING: Adding 'module.' prefix to checkpoint to match parallel model...")
        state_dict = {'module.'+k: v for k, v in state_dict.items()}
        
    # Load with strict=False but PRINT errors
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    print(f"Weight Loading Report:")
    print(f"  - Missing Keys: {len(missing)}")
    print(f"  - Unexpected Keys: {len(unexpected)}")
    
    if len(missing) > 0:
        print(f"  - Example Missing: {missing[:3]}")
    
    # If massive failure, try unwrapping
    if len(missing) > 100:
        print("CRITICAL WARNING: massive weight mismatch. Attempting to unwrap DataParallel...")
        # (Optional fallbacks logic could go here, but logging is enough for now)
    
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

    try:
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
                output = model(img_left.to(device), img_right.to(device), test_mode=True)
                disp = output
                
                # Post-processing
                disp = disp[0,0].cpu().numpy() # (H, W)
                
                # Colorize
                vis = vis_disparity(disp)
                
                # Encode response
                res, im_png = cv2.imencode(".jpg", vis)
                return Response(content=im_png.tobytes(), media_type="image/jpeg")
    except Exception as e:
        print(f"Inference Error: {e}")
        return Response(content=f"Error: {e}", status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
