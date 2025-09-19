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

    #### Figure out data capture limits ####

    # By detection split
    single_split = int(.6 * detection_limit)
    mlt_split = int(.2 * detection_limit)
    detectable_split = single_split + mlt_split
    neg_limit = int(.2 * detection_limit)

    small_total_split = int(.35 * detectable_split)
    med_total_split = int(.45 * detectable_split)
    big_total_split = int(.2 * detectable_split)

    # Combined detection and size split
    single_small_limit = int(.35 * single_split)
    single_med_limit = int(.45  * single_split)
    single_big_limit = int(.2  * single_split)

    mlt_small_limit = int(.35 * mlt_split)
    mlt_med_limit = int(.45  * mlt_split)
    mlt_big_limit = int(.2 * mlt_split)

    # Check math 
    check_total = (single_small_limit + single_med_limit + single_big_limit  + 
                    mlt_small_limit + mlt_med_limit + mlt_big_limit + 
                    neg_limit)

    check_single = single_small_limit + single_med_limit + single_big_limit
    check_mlt = mlt_small_limit + mlt_med_limit + mlt_big_limit


    if check_single != single_split:
        print("**************")
        print("issue with single split")
        print(f"Small Split: {single_small_limit}, Med Split: {single_med_limit}, Big Split: {single_big_limit}, Total: {single_small_limit + single_med_limit + single_big_limit}")
        print(f"Single Total: {single_split}")

    if check_mlt != mlt_split:
        print("**************")
        print("issue with Mltpl split")
        print(f"Small Split: {mlt_small_limit}, Med Split: {mlt_med_limit}, Big Split: {mlt_big_limit}")
        print(f"Single Total: {mlt_split}")

    if check_total != detection_limit:
        if (single_small_limit + mlt_small_limit) != (small_total_split):
            print("Something wrong with your SMALL splits")
            print(f"Single Split: {single_small_limit}, Mlt Split: {mlt_small_limit}, Total: {small_total_split}")
        elif (single_med_limit + mlt_med_limit) != (med_total_split):
            print("Something wrong with your MED splits")
            print(f"Single Split: {single_med_limit}, Mlt Split: {mlt_med_limit}, Total: {med_total_split}")
        elif (single_big_limit + mlt_big_limit) != (big_total_split):
            print("Something wrong with your BIG splits")
            print(f"Single Split: {single_big_limit}, Mlt Split: {mlt_big_limit}, Total: {big_total_split}")
        else:
            print("something is wrong but idk")
            print(f"Check Total: {check_total}, Detection Limit: {detection_limit}")
    else:
        print("good calc!")


    print(f"Small limit: {small_total_split}, Med limit: {med_total_split}, Big limit: {big_total_split}, Single limit: {single_split}, MLTp limit: {mlt_split}")

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
                dominant_detection = motion_filter.dominant_detection_size(small_total_split, med_total_split, big_total_split)
                should_save = False

                if count_category == "single" and (motion_filter.single_ctr < single_split):
                    if dominant_detection == "small" and small_single_ctr < single_small_limit:
                        should_save = True
                        small_single_ctr += 1
                        print(f'small single ctr: {small_single_ctr} / {small_total_split}')
                    elif dominant_detection == "med" and med_single_ctr < single_med_limit:
                        should_save = True
                        med_single_ctr += 1
                        print(f'med single ctr: {med_single_ctr} / {med_total_split}')
                    elif dominant_detection == "big" and big_single_ctr < single_big_limit:
                        should_save = True
                        big_single_ctr += 1
                        print(f'big single ctr: {big_single_ctr} / {big_total_split}')

                elif count_category == "multi" and (motion_filter.mltp_ctr < mlt_split):
                    if dominant_detection == "small" and small_multi_ctr < mlt_small_limit:
                        should_save = True
                        small_multi_ctr += 1
                        print(f'small multi ctr: {small_multi_ctr} / {small_total_split}')
                    elif dominant_detection == "med" and med_multi_ctr < mlt_med_limit:
                        should_save = True
                        med_multi_ctr += 1
                        print(f'med multi ctr: {med_multi_ctr} / {med_total_split}')
                    elif dominant_detection == "big" and big_multi_ctr < mlt_big_limit:
                        should_save = True
                        big_multi_ctr += 1
                        print(f'big multi ctr: {big_multi_ctr} / {big_total_split}')


                if should_save:
                    motion_filter.yolo_annotation(det_frame, True)
                    print('***************************')
                    print(f"Dominant {dominant_detection}")
                    print(f"Count Category: {count_category}")
                    print(f'Small Detections: {motion_filter.small_ctr} / {small_total_split}')
                    print(f'Small Detections: {small_single_ctr + small_multi_ctr} / {small_total_split}')
                    print(f'Med Detections: {motion_filter.med_ctr} / {med_total_split}')
                    print(f'Big Detections: {motion_filter.big_ctr} / {big_total_split}')
                    print("-")
                    print(f'Single Detections: {motion_filter.single_ctr} / {single_split}')
                    print(f'Mltpl Detections: {motion_filter.mltpl_ctr} / {mlt_split}')

                # Save negative training data
                # TODO: Instead of hidden class var, should use motion_filter return value to dictate
                # no_motion saving

                if motion_filter.neg_counter < neg_limit:
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

