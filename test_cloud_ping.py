import requests
import time

POD_ID = "isuny2rnbrmo7h"
URL = f"https://{POD_ID}-8888.proxy.runpod.net/"

print(f"Pinging {URL}...")
try:
    start = time.time()
    resp = requests.get(URL, timeout=5)
    dur = time.time() - start
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text}")
    print(f"Time: {dur:.3f}s")
except Exception as e:
    print(f"FAILED: {e}")
