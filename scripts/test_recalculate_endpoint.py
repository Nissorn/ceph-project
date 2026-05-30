import json
import requests
import sys
import time

URL = "http://localhost:8123/api/v1/recalculate"

payload = {
    "image_name": "test_image.jpg",
    "image_width": 512,
    "image_height": 512,
    "keypoints": [
        {"name": "Upper_tip", "x": 256.0, "y": 256.0, "confidence": 1.0},
        {"name": "Upper_apex", "x": 256.0, "y": 100.0, "confidence": 1.0},
        {"name": "Labial_midroot", "x": 200.0, "y": 200.0, "confidence": 1.0},
        {"name": "Labial_crest", "x": 200.0, "y": 250.0, "confidence": 1.0},
        {"name": "Palatal_midroot", "x": 300.0, "y": 200.0, "confidence": 1.0},
        {"name": "Palatal_crest", "x": 300.0, "y": 250.0, "confidence": 1.0},
        {"name": "ANS", "x": 400.0, "y": 300.0, "confidence": 1.0},
        {"name": "PNS", "x": 100.0, "y": 300.0, "confidence": 1.0},
        {"name": "LB", "x": 200.0, "y": 350.0, "confidence": 1.0},
        {"name": "PB", "x": 300.0, "y": 350.0, "confidence": 1.0}
    ],
    "polygons": [
        {
            "name": "Upper_incisor",
            "points": [250, 100, 260, 100, 260, 256, 250, 256, 250, 100]
        },
        {
            "name": "Labial_bone",
            "points": [150, 150, 200, 150, 200, 300, 150, 300, 150, 150]
        },
        {
            "name": "Palatal_bone",
            "points": [300, 150, 350, 150, 350, 300, 300, 300, 300, 150]
        }
    ]
}

print(f"Testing {URL} endpoint...")
max_retries = 15
for attempt in range(max_retries):
    try:
        response = requests.post(URL, json=payload, timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("SUCCESS! Response keys:", data.keys())
            print("Metrics:", json.dumps(data.get("data", {}).get("metrics", {}), indent=2))
            sys.exit(0)
        else:
            print("FAILED:", response.text)
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"Connection refused, waiting 2s... (attempt {attempt+1}/{max_retries})")
        time.sleep(2)
    except Exception as e:
        print("Error calling endpoint:", e)
        sys.exit(1)

print("Failed to connect to the backend server.")
sys.exit(1)
