# How to Run FoundationStereo on RunPod

Since your local machine lacks a powerful NVIDIA GPU, running on a cloud GPU (RunPod) is a great alternative.

## 1. Rent a Pod
1.  Go to [RunPod.io](https://www.runpod.io/).
2.  Click **Deploy** on a GPU instance (e.g., **RTX 3090** or **RTX 4090**).
3.  Choose a Template: **RunPod PyTorch 2.1** (or similar).
4.  Wait for the pod to start and click **Connect**.

## 2. Set Up the Project
You can use the **Web Terminal** or SSH.

First, clone your repository (since we pushed it earlier):
```bash
git clone https://github.com/SanderKeil/foundation-stereo-testing.git
cd foundation-stereo-testing
```

Then, run the setup script I created (copy the content below into a file named `setup.sh` on the pod, or just run these commands):

```bash
# Install dependencies
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
pip install -r requirements.txt
pip install gdown

# Download Weights
mkdir -p pretrained_models
cd pretrained_models
gdown --folder "https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing"
# Ensure the folder is named '23-51-11' inside pretrained_models
mv FoundationStereo_checkpoint/23-51-11 . || true
cd ..
```

## 3. Upload Your Captured Images
To safeguard the images you just captured locally, you need to upload them to the Pod.
In the JupyterLab interface (RunPod usually provides this on port 8888):
1.  Navigate to `foundation-stereo-testing/assets/`.
2.  Click the **Upload** button.
3.  Upload your local `my_left.png`, `my_right.png`, and `my_K.txt` (found in `assets/` on your computer).

## 4. Run the Demo
Now you can run the inference at full speed!

```bash
python scripts/run_demo.py \
  --left_file assets/my_left.png \
  --right_file assets/my_right.png \
  --intrinsic_file assets/my_K.txt \
  --ckpt_dir pretrained_models/23-51-11/model_best_bp2.pth \
  --out_dir output_runpod \
  --scale 1.0
```

## 5. Download Results
In JupyterLab, find the `output_runpod` folder, right-click the results (`vis.png`, `cloud.ply`), and **Download** them to your local machine to view in the 3D viewer.
