import json
import os
import sqlite3
from picamera2 import Picamera2
import cv2 as cv
import motion_class
import bird_model
import argparse
import datetime
import platform
from ultralytics import YOLO


def is_daytime(start_hour=6, start_minute=30, end_hour=19, end_minute=30):
    """Check if current time is within daytime hours"""
    now = datetime.datetime.now()
    current_time = now.time()
    
    # Create time objects for comparison
    start_time = datetime.time(start_hour, start_minute)
    end_time = datetime.time(end_hour, end_minute)
    
    return start_time <= current_time <= end_time

def hq_cam_trap():
    # Setup HQ Camera
    picam = Picamera2()
    config = picam.create_video_configuration(
        main={"size": (1920, 1080), "format": "RGB888"})
    picam.configure(config)
    picam.start()

    # Setup ML model
    os_name = platform.system()
    model_path = r'models\best_yolo11_birds_12062025.pt'
    detection_model = bird_model.BirdModel()
    detection_model.ML = True

    if os_name == "Windows":
        detection_model.model = YOLO(model_path)
    else:
        detection_model.ncnn_model(model_path) # on RPI will have to export as NCNN model !!!

    while True:
        frame = picam.capture_array()
        counter = counter + 1

        if counter > 30:
            # Show the re-sized webcam images
            orig_frame = frame.copy()
            gray = cv.cvtColor(orig_frame, cv.COLOR_RGB2GRAY)

            if detection_model.ML: 
                result = detection_model.model(orig_frame) 
                detection_model.parse_detection(result, frame)
                # get detection
                result = detection_model.model(orig_frame)
                # Parse detection
                sql_data = detection_model.parse_detection(result)
                if sql_data:
                    # Save annotated frame image
                    timestamp = int(datetime.datetime.now().timestamp() * 1000)  # milliseconds for uniqueness
                    img_file = f'detect_{timestamp}_{detection_model.downloads}.jpg'
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
        else:
            print("Waiting for background to stabilize")

        if cv.waitKey(20) & 0xFF == ord('d'): # stop looping, on videos after 20 miliseconds or when "d" is pressed
            break

    picam.stop()# closes video file
    cv.destroyAllWindows() # closes all windows

def main():
    parser = argparse.ArgumentParser(
        description='Arg parse for two vars - camera id and detection area'
    )

    # Run pipeline
    hq_cam_trap()

if __name__ == '__main__':
    main()

