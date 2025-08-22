import cv2 as cv
import time
from datetime import datetime
import os
from pathlib import Path

 # Set up camera
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Set to 1920x1080
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

# Verify settings
actual_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
print(f"Resolution set to: {actual_w}x{actual_h}")
print("Letting camera warm up...")

# Warm-up frames
for _ in range(3):
    cap.read()

# Setup time delta
delta = 300 # seconds, 5 mins
last_cap = 0
counter = 0
save_folder = Path('image/negative')
save_folder.mkdir(parents=True, exist_ok=True)

while True:
    flag, frame = cap.read()

    # Check if it's time to capture
    current_time = time.time()
    if current_time - last_cap >= delta:
        counter = counter + 1
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"negative_{timestamp}_{counter}.jpg"
        filepath = os.path.join(save_folder, filename) 
        cv.imwrite(filepath, frame)
        # Create empty text file for negative image
        neg_txt_file = f"negative_{timestamp}_{counter}.txt"
        neg_file_path = os.path.join(save_folder, neg_txt_file)
        Path(neg_file_path).touch()
        # Create empty text file to go with image
        last_cap = current_time
        print(f"saved random image: {counter}")

    if counter >= 120:
        print("Captured 120 possible negative images")
        break


