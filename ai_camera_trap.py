import sys
import time
from ultralytics import YOLO
import cv2
import utils
import ncnn
import numpy as np
from datetime import datetime
import sqlite3
import requests





def run(model='models/best_09082025.pt', camera_id=0, width=1920, height=1080) -> None:
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
        breakpoint()
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
