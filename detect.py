import argparse
import sys
import time
import subprocess
from ultralytics import YOLO
import os
import cv2
import utils
import ncnn
import numpy as np
from datetime import datetime
import sqlite3
import requests

def ncnn_model(model_path):
    # Load the YOLO11 model
    model = YOLO(model_path)

    # Export the model to NCNN format
    file,_ = os.path.splitext(model_path)
    expected_file_path = file +"_ncnn_model"
    if not os.path.exists(expected_file_path):
        model.export(format="ncnn")  # creates '/yolo11n_ncnn_model'

    # Load the exported NCNN model
    ncnn_model = YOLO(expected_file_path)

    return ncnn_model

def pre_process(img):
    img = cv2.flip(img, 1) # Because we are using a webcam
    image_resized = cv2.resize(img, (640,640))

    # TODO: Apply sunlight reduction based on time of day
    blur = cv2.GaussianBlur(image_resized, (5,5), cv2.BORDER_DEFAULT)
    rgb_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    return rgb_image


def import_db(metadata, filename):

    # Save metadata to SQLite database locally
    url = "http://192.168.0.159:8000/upload"

    with open(filename, "rb") as img_file:
        response = requests.post(
            url,
            files={"image": img_file},
            data=metadata
            )

    print(response.json())


def parse_detection(detection):

    detection = detection[0] 
    conf = detection.boxes.conf
    aId = detection.boxes.id
    cls = detection.boxes.cls
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
    totalDetections = cls.numel()
    clsList = cls.tolist()
    annotated_frame = detection.plot()

    if totalDetections > 0:
        print("detection")
        labels = [detection.names[x] for x in cls.tolist()]
        label = ", ".join(labels) 
        # TODO: How to store multiple confidence intervals?
        metadata = {
		"label": labels,
		"confidence": totalDetections,
                "timestamp": timestamp
		}

        file_name = label + "_" + timestamp + ".jpg"
        cv2.imwrite(file_name, annotated_frame)
        import_db(metadata, file_name) 

    return annotated_frame



def run(model='models/best.pt', camera_id=0, width=640, height=640) -> None:
    """Continuously run inference on images acquired from the camera.

    Args:
      model: Name of the TFLite object detection model.
      camera_id: The camera id to be passed to OpenCV.
      width: The width of the frame captured from the camera.
      height: The height of the frame captured from the camera.
    """

    # Variables to calculate FPS
    counter, fps = 0, 0
    start_time = time.time()

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Initialize the object detection model
    detector = ncnn_model(model)

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        counter += 1

        # Preprocess image
        rgb_image = pre_process(image)

	# Get Detection
        detection_result = detector(rgb_image)
        
        # Parse detection results
        annotated_frame = parse_detection(detection_result)
         
        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break
        #cv2.imshow('object_detector', annotated_frame)

    cap.release()
    cv2.destroyAllWindows()


def main():
    run()

if __name__ == '__main__':
    main()
