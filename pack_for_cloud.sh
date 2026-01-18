#!/bin/bash
# Zips the necessary files for the cloud
echo "Packing assets into assets.zip..."
zip -j assets.zip assets/my_left.png assets/my_right.png assets/my_K.txt
echo "Done! Upload 'assets.zip' to RunPod."
