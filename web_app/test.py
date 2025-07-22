# Import packages
import subprocess
from datetime import datetime
import sqlite3
import os
import requests


# Take a picture using USB Webcam
filename = 'img'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = filename + "_" + timestamp + ".jpg"
print(filename)
cmd = ["fswebcam", filename]
result = subprocess.run(cmd, check=True, capture_output=True, text = True)

# Extract metatdata from the image
metadata = {
    "label": "bird",
    "confidence": "0.9",
    "timestamp": timestamp
}

# Save metadata to SQLite database locally
url = "http://192.168.0.159:8000/upload"

with open(filename, "rb") as img_file:
    response = requests.post(
        url,
        files={"image": img_file},
        data=metadata
    )

print(response.json())

# Delete the image locally
os.remove(filename)
