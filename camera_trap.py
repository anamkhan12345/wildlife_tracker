import cv2 as cv
import pipeline_class
import argparse
import time

def camera_trap(cam_id, detection_area, detection_limit):

    # Define vegetation areas as rectangles (x1, y1, x2, y2)
    detector = pipeline_class.VegetationFilter()
    veg_zone = [
        (960, 0, 1919, 1079),
        (768, 930, 960, 1079),
        (800, 0, 1919, 756)
    ]

    # Motion filter over frames
    motion_filter = pipeline_class.MultipleFrameFilter(buffer_size=5, threshold=0.6)
    counter = 0

    # Set up camera
    cap = cv.VideoCapture(cam_id)
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

    # No motion saves
    delta = 180 # seconds, 3 mins


    while True:
        flag, frame = cap.read()
        counter = counter + 1

        if not flag or frame is None or frame.size == 0:
            print("Empty or invalid frame, skipping color conversion")
            breakpoint()
            break
        else:
            # Ignore first frames as camera turns on and background stabilizes
            if counter > 30:
                # Show the re-sized webcam images
                orig_frame = frame
                # Grid overlay
                grid_frame = pipeline_class.add_grid(orig_frame, rows=10, cols=10, thickness=1, alpha=0.5)

                # Check veg zone
                veg_plot_org = pipeline_class.plot_zone(orig_frame, veg_zone)
                txt = str(counter)
                cv.putText(veg_plot_org, txt, (0,1079), 
                        cv.FONT_HERSHEY_TRIPLEX, 0.5,
                        (0,255,0), 1, lineType=cv.LINE_AA)

                # Set Vegetation areas
                detector.set_vegetation_zones(orig_frame.shape, veg_zone)

                # Find motion
                motion = detector.adaptive_learning(orig_frame)

                # Filter for motion across multiple frames
                #filtered_frame = motion_filter.filter_motion(motion)

                # Filter motion found
                motion_filter.motion_filter(motion, orig_frame, detection_area, save_data=False)
                
                # Save negative training data
                motion_filter.no_motion_save(delta, orig_frame)

                # Display diffs
                #cv.imshow('Video', veg_plot_org)
                #cv.imwrite('image/filter.jpg', veg_plot_org)
                #cv.imshow('Grid Overlay', grid_frame)
                #cv.imwrite('image/grid.jpg', grid_frame)
                #cv.imshow('Vegetation Filter', motion)
            else:
                print("Waiting for background to stabilize")

        if cv.waitKey(20) & 0xFF == ord('d'): # stop looping, on videos after 20 miliseconds or when "d" is pressed
            break
        elif motion_filter.downloads > detection_limit:
            break

    cap.release() # closes video file
    cv.destroyAllWindows() # closes all windows

def main():
    parser = argparse.ArgumentParser(
        description='Arg parse for two vars - camera id and detection area'
    )
    parser.add_argument('-c', '--cam', type=int, default=0)
    parser.add_argument('-a', '--det_area', type=int, default=30)
    parser.add_argument('-l', '--det_cnt_limit', type=int, default=2000)

    # Parse arguments
    args = parser.parse_args()
    cam_id = args.cam
    min_area = args.det_area
    detection_limit = args.det_cnt_limit

    # Run pipeline
    camera_trap(cam_id, min_area, detection_limit)

if __name__ == '__main__':
    main()
