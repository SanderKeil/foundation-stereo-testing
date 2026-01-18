# Simple RunPod Guide 🚀

Runs FoundationStereo on a cloud GPU in 4 easy steps.

## 1. Local Prep
Run this on your computer terminal to pack your images:
```bash
bash pack_for_cloud.sh
```
*This creates `assets.zip`.*

## 2. Start Cloud Environment
1.  Go to [RunPod.io](https://www.runpod.io/).
2.  Deploy an **RTX 3090** or **4090** (Template: `RunPod PyTorch 2.1`).
3.  Click **Connect** -> **JupyterLab** -> **Terminal**.

## 3. Setup (One command)
Copy and paste this into the Cloud Terminal:
```bash
git clone https://github.com/SanderKeil/foundation-stereo-testing.git && cd foundation-stereo-testing && bash runpod_setup.sh
```
*Wait for it to finish installing.*

## 4. Run It
1.  **Drag & Drop** `assets.zip` from your computer into the JupyterLab file list (left sidebar).
2.  Run this in the terminal:
    ```bash
    bash run_cloud_job.sh
    ```
3.  **Right-click** `results.zip` and **Download**.

Done! Unzip `results.zip` on your computer to view the 3D model.
