
import cv2
import asyncio
import time
import requests
import aiohttp
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from scripts.capture_stereo import open_cameras_safe, set_camera_options

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Global Config
CLOUD_URL = "http://localhost:8000/process" # User will need to update this!
# We will effectively replace 'localhost' with the RunPod ID later.

# Global Camera State
left_cap = None
right_cap = None

def init_cameras():
    global left_cap, right_cap
    if left_cap and right_cap:
        return
        
    # Using the successful indices from previous tasks (6 and 4)
    # But better to use the robust finder or try knowns.
    # Let's hardcode the ones that worked: 6 (left/right?) and 4
    # capture_stereo.py said: 6 and 4.
    
    print("Initializing Cameras 6 and 4...")
    left_cap = cv2.VideoCapture(6)
    right_cap = cv2.VideoCapture(4)
    
    set_camera_options(left_cap)
    set_camera_options(right_cap)

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
def generate_camera_stream(camera_id):
    cap = left_cap if camera_id == "left" else right_cap
    while True:
        if cap is None: break
        ret, frame = cap.read()
        if not ret: break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/video_feed/{camera_id}")
async def video_feed(camera_id: str):
    return StreamingResponse(generate_camera_stream(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")


# The Magic Loop: Capture -> Send -> Result
@app.websocket("/ws/inference")
async def websocket_inference(websocket: WebSocket):
    await websocket.accept()
    
    # Re-init if needed
    init_cameras()
    
    try:
        while True:
            start_time = time.time()
            
            # 1. Capture Safe Pair
            # We want them synced.
            # Simple approach: grab sequentially fast.
            # Or use the 'grab' 'retrieve' method for sync.
            left_cap.grab()
            right_cap.grab()
            _, frame_left = left_cap.retrieve()
            _, frame_right = right_cap.retrieve()
            
            if frame_left is None or frame_right is None:
                await asyncio.sleep(0.1)
                continue
                
            # 2. Prepare for Upload
            _, left_jpg = cv2.imencode('.jpg', frame_left)
            _, right_jpg = cv2.imencode('.jpg', frame_right)
            
            files = {
                'left_image': ('left.jpg', left_jpg.tobytes(), 'image/jpeg'),
                'right_image': ('right.jpg', right_jpg.tobytes(), 'image/jpeg')
            }
            
            # 3. Send to Cloud (Async)
            # Need strict error handling here
            try:
                # Get the URL from a query param or global? 
                # Ideally pass it from client, but for now use global.
                # NOTE: aiohttp is better for async
                async with aiohttp.ClientSession() as session:
                    async with session.post(CLOUD_URL, data=files) as resp:
                        if resp.status == 200:
                            result_bytes = await resp.read()
                            # Convert to base64 or send raw bytes?
                            # Sending bytes over WS is efficient.
                            await websocket.send_bytes(result_bytes)
                        else:
                            print(f"Cloud Error: {resp.status}")
            except Exception as e:
                print(f"Connection Error: {e}")
                
            # Latency calc
            # elapsed = time.time() - start_time
            # print(f"Loop time: {elapsed:.2f}s")
            
            # Cap FPS to avoid overwhelming network?
            # await asyncio.sleep(0.1) 
            
    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
