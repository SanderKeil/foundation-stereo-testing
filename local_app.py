
import cv2
import asyncio
import time
import requests
import aiohttp
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Global Config
CLOUD_URL = "https://isuny2rnbrmo7h-8888.proxy.runpod.net/process" # User will need to update this!
# We will effectively replace 'localhost' with the RunPod ID later.

# Global Camera State
left_cap = None
right_cap = None
global_left_frame = None
global_right_frame = None
import threading
import threading
camera_lock = threading.Lock()
stop_camera_thread = False
camera_lock = threading.Lock()
stop_camera_thread = False
camera_thread = None

# Global Image Processing Parameters
global_vertical_shift = 0


def camera_loop():
    global global_left_frame, global_right_frame, left_cap, right_cap, stop_camera_thread
    print("Background Camera Loop Started")
    while not stop_camera_thread:
        if left_cap is None or right_cap is None:
            time.sleep(1)
            continue
            
        try:
             # Grab both first to sync as best as possible
             l_ret = left_cap.grab()
             r_ret = right_cap.grab()
             
             if l_ret and r_ret:
                 _, frame_left = left_cap.retrieve()
                 _, frame_right = right_cap.retrieve()
                 
                 if frame_left is not None: 
                     global_left_frame = frame_left
                 if frame_right is not None: 
                     global_right_frame = frame_right.copy()
             else:
                 # If grab fails, release and re-init? No, just wait.
                 time.sleep(0.1)
                 
        except Exception as e:
            print(f"Cam Error: {e}")
            time.sleep(0.5)
            
        time.sleep(0.01) # Max ~100 FPS cap

def set_camera_options(cap):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def init_cameras():
    global left_cap, right_cap, camera_thread
    if left_cap and right_cap:
        return

    print("State: Identifying Brio Cameras...")
    
    # Helper to try index list
    def open_cam(indices, name):
        for idx in indices:
            print(f"  > Attempting to open {name} at index {idx}...")
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                # --- LOCK COMMANDS ---
                # Attempt to disable Autofocus (0 = Off)
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) 
                
                # Attempt to lock White Balance (Auto=0). 
                # Note: Might need to set a specific temp, e.g. 4000.
                cap.set(cv2.CAP_PROP_AUTO_WB, 0)
                cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4000)
                
                # Test read
                ret, _ = cap.read()
                if ret:
                    print(f"    SUCCESS: {name} found at {idx}")
                    return cap
                else:
                    print(f"    WARNING: {name} at {idx} opened but failed to read frame.")
                    cap.release()
            else:
                 print(f"    Failed to open {idx}")
        return None

    try:
        if left_cap is None:
            # Try 0 then 1 for Brio 300
            left_cap = open_cam([0, 1], "Left (Brio 300)")
        
        if right_cap is None:
            # Try 6 then 7 for Brio 301
            right_cap = open_cam([6, 7], "Right (Brio 301)")
            
        if left_cap is None or right_cap is None:
            print("CRITICAL: Failed to open both cameras!")
        else:
            print("Cameras Initialized Successfully.")
            # START THREAD IF NOT RUNNING
            if camera_thread is None or not camera_thread.is_alive():
                camera_thread = threading.Thread(target=camera_loop, daemon=True)
                camera_thread.start()
            
    except Exception as e:
        print(f"Camera Open Error: {e}")

def get_latest_stereo_pair():
    if not left_cap or not right_cap:
        init_cameras()
        
    ret1, frame1 = left_cap.read()
    ret2, frame2 = right_cap.read()
    
    if not ret1 or not ret2:
        return None, None
        
    return frame1, frame2

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "cloud_url": CLOUD_URL})

# Stream Generator for Raw Video
# Stream Generator reading from Global Buffer
def generate_camera_stream(camera_id):
    while True:
        # Just peek at the global frame
        frame = global_left_frame if camera_id == "left" else global_right_frame
        
        if frame is None:
            time.sleep(0.1)
            continue
            
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033) # Limit to ~30 FPS polling

@app.get("/video_feed/{camera_id}")
async def video_feed(camera_id: str):
    return StreamingResponse(generate_camera_stream(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")


# --- NEW PARAMS ENDPOINT ---
from pydantic import BaseModel
class ShiftUpdate(BaseModel):
    shift: int

@app.post("/set_shift")
async def set_shift(update: ShiftUpdate):
    global global_vertical_shift
    global_vertical_shift = update.shift
    print(f"Update: Vertical Shift set to {global_vertical_shift}")
    return {"status": "ok", "shift": global_vertical_shift}


# The Magic Loop: Capture -> Send -> Result
@app.websocket("/ws/inference")
async def websocket_inference(websocket: WebSocket):
    global global_left_frame, global_right_frame 
    await websocket.accept()
    
    debug_save_count = 0  # Counter for debug saves
    
    # Send initial status
    await websocket.send_text("STATUS: Connected. Initializing cameras...")
    
    # Auto-init if needed (blocking but protected)
    if not left_cap or not right_cap:
        try:
             # Run scanner in thread to avoid blocking heartbeat
             await asyncio.to_thread(init_cameras)
        except Exception as e:
             await websocket.send_text(f"STATUS: Camera Init Failed: {str(e)}")
    
    if not left_cap or not right_cap:
         await websocket.send_text("STATUS: No Cameras Found! Check USB.")
         # Don't crash, just wait, so user sees the error
         try:
             while True:
                 await asyncio.sleep(1)
         except WebSocketDisconnect:
             return
    
    await websocket.send_text("STATUS: Cameras Ready. Starting Stream...")

    try:
        while True:
            # 1. READ GLOBALS ONLY
            if global_left_frame is None or global_right_frame is None:
                 await asyncio.sleep(0.1)
                 continue

            frame_left = global_left_frame
            frame_right = global_right_frame
            
            # --- APPLY MANUAL RECTIFICATION (Vertical Shift) ---
            if global_vertical_shift != 0:
                # Shift Right Image Up/Down to match Left
                M = np.float32([[1, 0, 0], [0, 1, global_vertical_shift]])
                # Use borderReplicate to avoid black bars messing up disparity
                frame_right = cv2.warpAffine(frame_right, M, (frame_right.shape[1], frame_right.shape[0]), borderMode=cv2.BORDER_REPLICATE)
                
                
            # 2. Prepare for Upload - Use PNG for MAX QUALITY (Lossless)
            # Latency will be high, but "fuzziness" will be gone.
            _, left_png = cv2.imencode('.png', frame_left)
            _, right_png = cv2.imencode('.png', frame_right)
            
            # Construct FormData for aiohttp
            data = aiohttp.FormData()
            data.add_field('left_image', left_png.tobytes(), filename='left.png', content_type='image/png')
            data.add_field('right_image', right_png.tobytes(), filename='right.png', content_type='image/png')
            
            # DEBUG: Save inputs to prove they are valid
            if debug_save_count < 3:
                 cv2.imwrite(f"debug_sent_left_{debug_save_count}.jpg", frame_left)
                 cv2.imwrite(f"debug_sent_right_{debug_save_count}.jpg", frame_right)
                 print(f"DEBUG: Saved debug_sent_{debug_save_count}.jpg")
                 debug_save_count += 1
            
            # 3. Send to Cloud (Async)
            try:
                # req_start = time.time()
                async with aiohttp.ClientSession() as session:
                    async with session.post(CLOUD_URL, data=data) as resp:
                        # req_dur = time.time() - req_start
                        
                        if resp.status == 200:
                            result_bytes = await resp.read()
                            # Check if binary or text error
                            if len(result_bytes) > 1000:
                                await websocket.send_bytes(result_bytes)
                                # print(f"Success! Round trip: {req_dur:.3f}s")
                            else:
                                print(f"Error: Response too small ({len(result_bytes)}B): {result_bytes}")
                        else:
                            print(f"Cloud Error: {resp.status}")
                            # await websocket.send_text(f"STATUS: Cloud Error {resp.status}")
            except Exception as e:
                print(f"Connection Error: {e}")
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"CRITICAL WEBSOCKET ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
