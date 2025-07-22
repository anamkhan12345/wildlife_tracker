# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
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


def basic_model(model_path):
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

    # Visualization parameters
    row_size = 20  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    # Initialize the object detection model
    detector = basic_model(model)

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        counter += 1
        image = cv2.flip(image, 1)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# Get Detection
        detection_result = detector(rgb_image)


        # Display frame
        annotated_frame = detection_result[0].plot()
 
        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('object_detector', annotated_frame)

    cap.release()
    cv2.destroyAllWindows()


def main():
    run()

if __name__ == '__main__':
    main()
