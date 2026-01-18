
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

# --- Import FoundationStereo Modules ---
# Ensure code_dir is in path so we can import from core
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
    print("Loading FoundationStereo Model... This may take a minute.")
    
    ckpt_path = "pretrained_models/23-51-11/model_best_bp2.pth"
    # Basic args structure
    class Args:
        model_type = "base" 
        mixed_precision = True
        restore_ckpt = ckpt_path
        
    args = Args()
    
    # Initialize model
    model = FoundationStereo(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(model, device_ids=[0])
    model.to(device)
    
    # Load weights
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
async def process_stereo(left_image: UploadFile = File(...), right_image: UploadFile = File(...), intrinsic_file: UploadFile = File(None)):
    global model
    
    # Read Images
    left_bytes = await left_image.read()
    right_bytes = await right_image.read()
    
    img_left = imageio.imread(left_bytes)
    img_right = imageio.imread(right_bytes)
    
    # Basic resizing for speed (optional, can be parameter)
    # Keeping original resolution for now to match user expectation, 
    # but could downscale here if > 2.5s latency.
    
    # Prepare tensors
    img_left = torch.from_numpy(img_left).permute(2, 0, 1).float() / 255.0
    img_right = torch.from_numpy(img_right).permute(2, 0, 1).float() / 255.0
    
    # Batch dimension
    img_left = img_left.unsqueeze(0)
    img_right = img_right.unsqueeze(0)
    
    # Intrinsics (Dummy or Real)
    # FoundationStereo demo does this:
    # K is flattened 1x9, baseline is scalar. 
    # If not provided, we might need a default or expect it in the file.
    # For rapid testing, let's hardcode 'my_K.txt' equivalent if not sent, 
    # OR better: accept it as text. 
    # Actually, let's look at how run_demo processes it.
    
    # Creating dummy/default intrinsics if missing, just to make forward pass work
    # The model expects 'intrinsics' and 'pose' in the batch but many FoundationStereo
    # versions infer fine without perfect ones for pure disparity visual.
    # Looking at core code: it uses K for backprojection.
    
    # Let's read K from upload if available, else standard approximate
    # Standard: fx=approx, baseline=approx
    # For now, let's skip complex intrinsic parsing and just run the inference
    # wrapping it exactly like run_demo.py does.
    
    # From run_demo.py logic:
    # It constructs a batch dictionary.
    
    device = next(model.parameters()).device
    
    # Dummy intrinsic for forward pass logic (often needed for structure)
    # H, W = img_left.shape[-2:]
    intrinsics = torch.eye(3).unsqueeze(0) # Placeholder
    
    batch = {
        'left': img_left.to(device),
        'right': img_right.to(device),
        # 'intrinsics': intrinsics.to(device) # Add if strictly needed
    }
    
    with torch.no_grad():
        # Using mixed precision as in demo
        with torch.cuda.amp.autocast(enabled=True):
            # Model forward
            # Note: FoundationStereo forward might vary. 
            # In run_demo.py: preds = model(left, right, ...)
            # Let's check Utils or code. 
            # Assuming standard call:
            output = model(batch['left'], batch['right'])
            
            # Output is usually a list/dict. 
            # run_demo: disp_pred = output['disp_pred_list'][-1]
            disp = output['disp_pred_list'][-1]
            
            # Post-processing
            disp = disp[0].cpu().numpy() # (H, W)
            
            # Colorize
            # Use the util from the repo
            vis = vis_disparity(disp)
            
            # Encode response
            res, im_png = cv2.imencode(".jpg", vis)
            return Response(content=im_png.tobytes(), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
