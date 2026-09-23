import json
import urllib.request

payload = {f"feature_{i}": 0.0 for i in range(1, 167)}

req = urllib.request.Request(
    "http://localhost:8000/predict",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)

with urllib.request.urlopen(req) as response:
    print(response.read().decode())