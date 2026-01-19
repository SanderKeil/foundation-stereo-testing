
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
camera_lock = threading.Lock()

def set_camera_options(cap):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def init_cameras():
    global left_cap, right_cap
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
            # 1. Capture Safe Pair
            if left_cap is None or right_cap is None:
                 await asyncio.sleep(1)
                 continue

            try:
                # Thread-safe read?
                # Actually, since we are the MAIN loop, we own the camera.
                # But we should update the globals for the other endpoint.
                
                left_cap.grab()
                right_cap.grab()
                _, frame_left = left_cap.retrieve()
                _, frame_right = right_cap.retrieve()
                
                # Update globals for the video feed
                if frame_left is not None: 
                    global_left_frame = frame_left
                    # print(f"DEBUG: Captured Left {frame_left.shape}")
                    # print(f"DEBUG: Encoded sizes: L={len(left_jpg)} R={len(right_jpg)}")
                if frame_right is not None: 
                    global_right_frame = frame_right
                
            except Exception as e:
                print(f"Capture Error: {e}")
                await asyncio.sleep(0.5)
                continue
            
            if frame_left is None or frame_right is None:
                # print("Skipping dropped frame")
                await asyncio.sleep(0.01)
                continue
                
            # 2. Prepare for Upload
            _, left_jpg = cv2.imencode('.jpg', frame_left)
            _, right_jpg = cv2.imencode('.jpg', frame_right)
            
            # Construct FormData for aiohttp
            data = aiohttp.FormData()
            data.add_field('left_image', left_jpg.tobytes(), filename='left.jpg', content_type='image/jpeg')
            data.add_field('right_image', right_jpg.tobytes(), filename='right.jpg', content_type='image/jpeg')
            
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
