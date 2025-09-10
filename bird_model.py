import os
import cv2 as cv
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import time
import requests

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


    def parse_detection(self, detection_result, original_frame):

        detection = detection_result[0]
        boxes = [x.xywh.tolist() for x in detection.boxes]
        areas = [x[0][2] * x[0][3] for x in boxes]
        max_area = int(max(areas)) if areas else 0

        conf = detection.boxes.conf
        aId = detection.boxes.id
        cls = detection.boxes.cls
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
        totalDetections = cls.numel()
        clsList = cls.tolist()
        annotated_frame = detection.plot()
        self.detection = False
        if totalDetections > 0:
            print("detection")
            breakpoint()
            self.detection = True
            self.downloads += 1
            labels = [detection.names[x] for x in cls.tolist()]
            label = ", ".join(labels) 
            # TODO: How to store multiple confidence intervals?
            metadata = {
            "label": labels,
            "confidence": totalDetections,
                    "timestamp": timestamp
            }
            timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
            img_file = f'detect_{timestamp}_area_{max_area}_{self.downloads}.jpg'
            raw_file = f'raw_{timestamp}_area_{max_area}_{self.downloads}.jpg'
            cv.imwrite(img_file, annotated_frame)
            cv.imwrite(raw_file, original_frame)
            #import_db(metadata, file_name)
        else:
            pass

        return annotated_frame


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
