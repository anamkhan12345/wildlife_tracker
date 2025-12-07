import sqlite3
import os
from turtle import pd
import cv2 as cv
from ultralytics import YOLO
import numpy as np
from datetime import date, datetime
import time
import requests
import pandas as pd

class BirdModel:
    def __init__(self):
        self.downloads = 0
        self.detection = False
        self.ML = True
        self.model = None

    def ncnn_model(self, model_path):
        # Load the YOLO11 model
        model = YOLO(model_path)

        # Export the model to NCNN format
        file,_ = os.path.splitext(model_path)
        expected_file_path = file +"_ncnn_model"
        if not os.path.exists(expected_file_path):
            model.export(format="ncnn")  # creates '/yolo11n_ncnn_model'

        # Load the exported NCNN model
        ncnn_model = YOLO(expected_file_path)
        self.model = ncnn_model


    def pre_process(self, img):
        #img = cv.flip(img, 1) # Because we are using a webcam
        #image_resized = cv.resize(img, (640,640))

        # TODO: Apply sunlight reduction based on time of day
        rgb_image = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        return rgb_image


    def parse_detection(self, detection_result, original_frame=None):

        detection = detection_result[0]
        cls = detection.boxes.cls
        totalDetections = cls.numel()
        self.detection = False
        detection_info = {}
        if totalDetections > 0:
            print(f"Detections: {totalDetections}")
            # Areas
            boxes = [x.xywh.tolist() for x in detection.boxes]
            areas = [x[0][2] * x[0][3] for x in boxes]
            max_area = int(max(areas)) if areas else 0
            # Confidence Score 
            conf = detection.boxes.conf
            conf_metadata = conf.tolist()
            # Timestamp and img array
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            hour_only = time.strftime("%H", time.localtime())
            # Annotated Frame
            annotated_frame = detection.plot()

            # Set up dictionary to return
            detection_info = {'timestamp': timestamp,
                        'hour_of_day': hour_only,
                        'total_detections': totalDetections,
                        'max_area': max_area,
                        'detection_json': [],
                        'annotated_frame': annotated_frame,
                        }
            self.detection = True
            self.downloads += 1
            labels = [detection.names[x] for x in cls.tolist()]

            # Create a JSON array for SQLite to handle
            for i in range(0, totalDetections):
                json_metadata = {
                    "label": labels[i],
                    "confidence": conf_metadata[i],
                    "area": areas[i]
                }
                detection_info['detection_json'].append(json_metadata)

        else:
            pass

        return detection_info


    def import_db(self, metadata, filename):

        # Save metadata to SQLite database locally
        url = "http://192.168.0.159:8000/upload"

        with open(filename, "rb") as img_file:
            response = requests.post(
                url,
                files={"image": img_file},
                data=metadata
                )

        print(response.json())


    def get_all_dashboard_stats(self):
        conn = sqlite3.connect('detections.db')
        cursor = conn.cursor()
        
        # 1. Total all time
        cursor.execute("SELECT SUM(detection_count) FROM detections")
        total_all_time = cursor.fetchone()[0] or 0
        
        # 2. Total today
        cursor.execute("""
            SELECT SUM(detection_count) 
            FROM detections 
            WHERE DATE(timestamp) = DATE('now')
        """)
        total_today = cursor.fetchone()[0] or 0
        
        # 3. Hourly averages (all time)
        cursor.execute("""
            SELECT 
                hour_of_day,
                AVG(detection_count) as avg_detections
            FROM detections
            GROUP BY hour_of_day
            ORDER BY CAST(hour_of_day AS INTEGER)
        """)
        hourly_data = cursor.fetchall()
        hourly_df = pd.DataFrame(hourly_data, columns=['Hour', 'Avg Detections'])
        
        # 4. Latest image
        cursor.execute("""
            SELECT image_path 
            FROM detections 
            ORDER BY timestamp DESC 
            LIMIT 1
        """)
        latest_result = cursor.fetchone()
        latest_image_path = latest_result[0] if latest_result else None
        
        # 5. Largest detection of all time
        cursor.execute("""
            SELECT image_path, max_detection_area
            FROM detections 
            ORDER BY max_detection_area DESC 
            LIMIT 1
        """)
        largest_result = cursor.fetchone()
        largest_image_path = largest_result[0] if largest_result else None
        largest_area = largest_result[1] if largest_result else 0
        
        conn.close()
        
        return {
            'total_all_time': total_all_time,
            'total_today': total_today,
            'hourly_df': hourly_df,
            'latest_image': latest_image_path,
            'largest_image': largest_image_path,
            'largest_area': largest_area
        }
    


