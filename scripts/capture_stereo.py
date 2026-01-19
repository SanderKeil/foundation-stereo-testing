import cv2
import time
import sys

def capture_camera(index, filename):
    print(f"Capturing from camera {index} to {filename}...")
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Failed to open camera {index}")
        return False
    
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Warmup
    for _ in range(15):
        cap.read()
        time.sleep(0.05)
        
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
    else:
        print(f"Failed to read from camera {index}")
        
    cap.release()
    return ret

def capture_stereo(device_id_left, device_id_right, output_dir="assets"):
    print(f"Attempting SYNCHRONIZED capture from {device_id_left} and {device_id_right}...")
    
    # Strategy: Open 1, Set MJPG (low bandwidth), Then Open 2
    
    # 1. Open Left
    print(f"Opening camera {device_id_left}...")
    cap1 = cv2.VideoCapture(device_id_left, cv2.CAP_V4L2)
    if not cap1.isOpened():
        print(f"Failed to open camera {device_id_left}")
        return False
    # Set MJPG *immediately* to minimize bandwidth usage, hoping it frees space for the second camera
    cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 2. Open Right
    print(f"Opening camera {device_id_right}...")
    cap2 = cv2.VideoCapture(device_id_right, cv2.CAP_V4L2)
    if not cap2.isOpened():
        print(f"Failed to open camera {device_id_right} (likely bandwidth limit)")
        cap1.release()
        return False
    cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Warmup both
    print("Warming up...")
    for _ in range(10):
        cap1.grab()
        cap2.grab()
        # Retrieve to clear buffer
        cap1.retrieve()
        cap2.retrieve()
        
    # Capture "Sync"
    print("Capturing...")
    cap1.grab()
    cap2.grab()
    ret1, frame1 = cap1.retrieve()
    ret2, frame2 = cap2.retrieve()
    
    cap1.release()
    cap2.release()
    
    if ret1 and ret2:
        # Force resize to match expected dimensions
        frame1 = cv2.resize(frame1, (640, 480))
        frame2 = cv2.resize(frame2, (640, 480))
        cv2.imwrite(f"{output_dir}/my_left.png", frame1)
        cv2.imwrite(f"{output_dir}/my_right.png", frame2)
        print("Synchronized capture saved.")
        return True
    else:
        print("Failed to retrieve frames.")
        return False

if __name__ == "__main__":
    # Identified mapping:
    # video6: BRIO 300 
    # video4: Brio 301
    
    # Prioritize 6 and 4 as they were the ones locked by Chrome.
    output_path = "../assets"
    if capture_stereo(6, 4, output_dir=output_path):
        print("Success capturing from 6 and 4.")
        sys.exit(0)
        
    if capture_stereo(7, 5):
        print("Success capturing from 7 and 5.")
        sys.exit(0)
    
    print("Direct capture failed. Scanning all devices...")
    valid_indices = []
    for i in range(12):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            # Try to force MJPG/640x480 during scan to handle bandwidth
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, frame = cap.read()
            if ret:
                print(f"Found working camera at index {i}")
                valid_indices.append(i)
            cap.release()
            
    print(f"Valid indices found: {valid_indices}")
    # Brio 301 is likely 5. Brio 300 MUST be one of the others?
    # If we only find 2 (Internal) and 5 (Brio 301), then Brio 300 is dead/locked.
    
    if len(valid_indices) >= 2:
        # User wants Brio 300 + Brio 301.
        # If we have [2, 5], that's Internal + Brio 301.
        # If we have [5, X], we assume X is the other Brio?
        # Let's try to pick 5 and the other non-2 index if possible.
        
        cams = [x for x in valid_indices if x != 2] # Exclude internal if possible
        if len(cams) >= 2:
            print(f"Capturing from {cams[0]} and {cams[1]}")
            capture_stereo(cams[0], cams[1], output_dir=output_path)
        elif len(cams) == 1 and 2 in valid_indices:
             print(f"Could not find two Brios. Capturing from {cams[0]} (Brio?) and 2 (Internal).")
             capture_stereo(cams[0], 2, output_dir=output_path)
        else:
             print("Not enough cameras found.")

