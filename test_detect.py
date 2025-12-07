from datetime import date
import json
import os
import time
import cv2 as cv
import glob
from ultralytics import YOLO
import bird_model
import sqlite3
import streamlit as st



# Import the detection module
model_path = r'C:\Users\anamk\projects\dataSets\models\yolo11_birds_12062025.pt'
model = YOLO(model_path) # on RPI will have to export as NCNN model !!!

# Set file path for folder with 5 images
test_dir = r'C:\Users\anamk\projects\dataSets\birds.v4i.yolov11'

# Get all image files in that folder
test_files = glob.glob(f"{test_dir}/**/*", recursive=True)
test_files = [f for f in test_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
model_utils = bird_model.BirdModel()

# loop over those images
for file in test_files:
    # get detection
    result = model(file)
    # Parse detection
    sql_data = model_utils.parse_detection(result)
    if sql_data:
        # Save annotated frame image
        timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
        img_file = f'detect_{timestamp}_{model_utils.downloads}.jpg'
        detection_dir = r'C:\Users\anamk\projects\wildlife_tracker\detection_images'
        if not os.path.exists(detection_dir):
            os.mkdir(detection_dir)
        img_file = f'{detection_dir}\{img_file}'
        cv.imwrite(img_file, sql_data['annotated_frame'])
        # Send info to SQLite DB
        conn = sqlite3.connect("detections.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO detections (timestamp,  hour_of_day, image_path, detection_count, max_detection_area, detection_data)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (sql_data['timestamp'],
            sql_data['hour_of_day'],
            img_file,
            sql_data['total_detections'], 
            sql_data['max_area'], 
            json.dumps(sql_data['detection_json'][0])) )
        conn.commit()
        conn.close()
    else: 
        print("No detections found.")
