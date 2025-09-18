from picamera2 import Picamera2
import cv2 as cv
import motion_class
import bird_model
import argparse
import datetime

def is_daytime(start_hour=6, start_minute=30, end_hour=19, end_minute=30):
    """Check if current time is within daytime hours"""
    now = datetime.datetime.now()
    current_time = now.time()
    
    # Create time objects for comparison
    start_time = datetime.time(start_hour, start_minute)
    end_time = datetime.time(end_hour, end_minute)
    
    return start_time <= current_time <= end_time

def hq_cam_trap(cam_id, detection_area, detection_limit):
    # Setup HQ Camera
    picam = Picamera2()
    config = picam.create_video_configuration(
        main={"size": (1920, 1080), "format": "RGB888"})
    picam.configure(config)
    picam.start()
    
    # Define vegetation areas as rectangles (x1, y1, x2, y2)
    detector = motion_class.VegetationFilter()
    veg_zone = [
            (960, 0, 1919, 1079),
            (768, 756, 960, 1079)
        ]

    # Motion filter over frames
    motion_filter = motion_class.MultipleFrameFilter(buffer_size=5, threshold=0.6)
    motion_filter.debug = False 
    counter = 0

    # No motion saves
    delta = 180 # seconds, 5 mins

    # Setup ML model
    model_path = r'models/best_09082025.pt'
    detection_model = bird_model.BirdModel()
    detection_model.ncnn_model(model_path)
    detection_model.ML = False

    # Figure out data capture limits
    small_limit = int(detection_limit * 0.3)
    med_limit = int(detection_limit * 0.45)
    big_limit = int(detection_limit * 0.25)
    single_limit = int(detection_limit * .45)
    mltp_limit = int(detection_limit * .4)
    dense_limit = int(detection_limit * .15)
    print(f"Small limit: {small_limit}, Med limit: {med_limit}, Big limit: {big_limit}, Single limit: {single_limit}, MLTp limit: {mltp_limit}, Dense limit: {dense_limit}")

    # Counters for combined categories (add these)
    small_single_ctr = 0
    small_multi_ctr = 0
    med_single_ctr = 0
    med_multi_ctr = 0
    big_single_ctr = 0
    big_multi_ctr = 0

    while True:
        frame = picam.capture_array()
        counter = counter + 1

        if not is_daytime():
            print(f"Outside daytime hours, skipping frame at {datetime.datetime.now().strftime('%H:%M:%S')}")
            continue  # This jumps back to the top of the while loop

        if counter > 30:
            # Show the re-sized webcam images
            orig_frame = frame.copy()
            gray = cv.cvtColor(orig_frame, cv.COLOR_RGB2GRAY)
            blur = cv.GaussianBlur(gray, (5,5),0)

            if detection_model.ML:
                result = detection_model.model(orig_frame) 
                detection_model.parse_detection(result, frame)
            else:
                # Grid overlay
                grid_frame = motion_class.add_grid(orig_frame, rows=10, cols=10, thickness=1, alpha=0.5)
                
                # Check veg zone
                veg_plot_org = motion_class.plot_zone(orig_frame, veg_zone)
                txt = str(counter)
                cv.putText(veg_plot_org, txt, (0,1079), 
                           cv.FONT_HERSHEY_TRIPLEX, 0.5,
                           (0,255,0), 1, lineType=cv.LINE_AA)

                # Set Vegetation areas
                #detector.set_vegetation_zones(orig_frame.shape, veg_zone)

                # Find motion
                motion = detector.adaptive_learning(gray)

                # Filter motion found
                det_frame = motion_filter.motion_filter(motion, orig_frame, detection_area)

                # Data collection scheduling
                count_category = motion_filter.detection_category
                priority = motion_filter.prioritize_data_collect()
                should_save = False

                if count_category == "single":
                    if priority == "small" and small_single_ctr < int((small_limit * 0.6)):  # 60% of small should be single
                        should_save = True
                        small_single_ctr += 1
                    elif priority == "med" and med_single_ctr < int((med_limit * 0.7)):  # 70% of med should be single
                        should_save = True
                        med_single_ctr += 1
                    elif priority == "big" and big_single_ctr < int((big_limit * 0.8)):  # 80% of big should be single
                        should_save = True
                        big_single_ctr += 1

                elif count_category == "multi":
                    if priority == "small" and small_multi_ctr < (small_limit * 0.4):
                        should_save = True
                        small_multi_ctr += 1
                    elif priority == "med" and med_multi_ctr < (med_limit * 0.3):
                        should_save = True
                        med_multi_ctr += 1
                    elif priority == "big" and big_multi_ctr < (big_limit * 0.2):
                        should_save = True
                        big_multi_ctr += 1

                elif count_category == "dense":
                    # Dense scenes are valuable regardless of size, but cap them
                    if dense_ctr < dense_limit:
                        should_save = True
                        dense_ctr += 1

                if should_save:
                    motion_filter.yolo_annotation(det_frame, True)
                    print('***************************')
                    print(f'Small Detections: {motion_filter.small_ctr} / {small_limit}')
                    print(f'Med Detections: {motion_filter.med_ctr} / {med_limit}')
                    print(f'Big Detections: {motion_filter.big_ctr} / {big_limit}')
                    print("-")
                    print(f'Single Detections: {motion_filter.single_ctr} / {single_limit}')
                    print(f'Mltpl Detections: {motion_filter.mltpl_ctr} / {single_limit}')
                    print(f'Dense Detections: {motion_filter.dense_ctr} / {single_limit}')

                # Save negative training data
                # TODO: Instead of hidden class var, should use motion_filter return value to dictate
                # no_motion saving
                motion_filter.no_motion_save(delta, orig_frame)

                # Display diffs
                #cv.imshow('Video', orig_frame)
                #cv.imshow('Gray', gray)
                #cv.imshow('Blur', blur)
                #cv.imwrite('image/filter.jpg', veg_plot_org)
                #cv.imshow('Grid Overlay', grid_frame)
                #cv.imwrite('image/grid.jpg', grid_frame)
                #cv.imshow('Vegetation Filter', motion)
                #cv.imshow('Motion Filter', filtered_frame)
        else:
            print("Waiting for background to stabilize")

        if cv.waitKey(20) & 0xFF == ord('d'): # stop looping, on videos after 20 miliseconds or when "d" is pressed
            break
        elif motion_filter.downloads > detection_limit or detection_model.downloads > detection_limit:
            break

    picam.stop()# closes video file
    cv.destroyAllWindows() # closes all windows

def main():
    parser = argparse.ArgumentParser(
        description='Arg parse for two vars - camera id and detection area'
    )
    parser.add_argument('-c', '--cam', type=int, default=0)
    parser.add_argument('-a', '--det_area', type=int, default=55)
    parser.add_argument('-l', '--det_cnt_limit', type=int, default=200)

    # Parse arguments
    args = parser.parse_args()
    cam_id = args.cam
    min_area = args.det_area
    detection_limit = args.det_cnt_limit

    # Run pipeline
    hq_cam_trap(cam_id, min_area, detection_limit)

if __name__ == '__main__':
    main()

