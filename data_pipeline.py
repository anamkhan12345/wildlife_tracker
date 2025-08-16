import cv2 as cv
import pipeline_class
import argparse
import time

def pipeline(cam_id, detection_area, detection_limit):

    # Define vegetation areas as rectangles (x1, y1, x2, y2)
    detector = pipeline_class.VegetationFilter()
    veg_zone = [
        (470, 0, 639, 639),
        (370, 440, 520, 639)
        # (0,0,60,470)
    ]

    # Motion filter over frames
    motion_filter = pipeline_class.MultipleFrameFilter(buffer_size=5, threshold=0.6)
    counter = 0

    cap = cv.VideoCapture(cam_id)
    print("Letting camera warm up...")

    # Warm-up frames
    for _ in range(3):
        cap.read()

    while True:
        flag, frame = cap.read()
        counter = counter + 1

        if not flag or frame is None or frame.size == 0:
            print("Empty or invalid frame, skipping color conversion")
            breakpoint()
            break
        else:
            # Ignore first frames as camera turns on and background stabilizes
            if counter > 60:
                # Show the re-sized webcam images
                orig_frame = cv.resize(frame, (640,640), interpolation=cv.INTER_AREA)

                # Grid overlay
                grid_frame = pipeline_class.add_grid(orig_frame, rows=10, cols=10, thickness=1, alpha=0.5)

                # Check veg zone
                veg_plot_org = pipeline_class.plot_zone(orig_frame, veg_zone)
                txt = str(counter)
                cv.putText(veg_plot_org, txt, (0,639), 
                        cv.FONT_HERSHEY_TRIPLEX, 0.5,
                        (0,255,0), 1, lineType=cv.LINE_AA)

                # Set Vegetation areas
                detector.set_vegetation_zones(orig_frame.shape, veg_zone)

                # Find motion
                motion = detector.adaptive_learning(orig_frame)

                # Filter for motion across multiple frames
                filtered_frame = motion_filter.filter_motion(motion)

                # Save any groups found
                detection = motion_filter.annotate_data(motion, orig_frame, detection_area)

                # Display diffs
                cv.imshow('Video', veg_plot_org)
                # cv.imshow('Grid Overlay', grid_frame)
                cv.imshow('Vegetation Filter', motion)
                #cv.imshow('Motion Filter', filtered_frame)
            else:
                print("Waiting for background to stabilize")

        if cv.waitKey(20) & 0xFF == ord('d'): # stop looping on videos after 20 miliseconds or when "d" is pressed
            break
        elif motion_filter.downloads > detection_limit:
            break

    cap.release() # closes video file
    cv.destroyAllWindows() # closes all windows

def main():
    parser = argparse.ArgumentParser(
        description='Arg parse for two vars - camera id and detection area'
    )
    parser.add_argument('-c', '--cam', type=int, default=1)
    parser.add_argument('-a', '--det_area', type=int, default=20)
    parser.add_argument('-l', '--det_limit', type=int, default=200)

    # Parse arguments
    args = parser.parse_args()
    cam_id = args.cam
    detection_area = args.det_area
    detection_limit = args.det_limit

    # Run pipeline
    pipeline(cam_id, detection_area, detection_limit)

if __name__ == '__main__':
    main()