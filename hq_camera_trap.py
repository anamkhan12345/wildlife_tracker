from picamera2 import Picamera2
import cv2 as cv
import pipeline_class
import argparse
import time

cam = Picamera2()
print(f"Camera modes: {cam.sensor_modes}")

# Define vegetation areas as rectangles (x1, y1, x2, y2)
detector = pipeline_class.VegetationFilter()
veg_zone = [
        (960, 0, 1919, 1079),
        (768, 756, 960, 1079)
    ]

# Motion filter over frames
motion_filter = pipeline_class.MultipleFrameFilter(buffer_size=5, threshold=0.6)
counter = 0

# No motion saves
delta = 300 # seconds, 5 mins

while True:
    frame = picam.capture_array()
    counter = counter + 1
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
        #detector.set_vegetation_zones(orig_frame.shape, veg_zone)

        # Find motion
        motion = detector.adaptive_learning(orig_frame)

        # Filter for motion across multiple frames
        #filtered_frame = motion_filter.filter_motion(motion)

        # Filter motion found
        detection = motion_filter.motion_filter(motion, orig_frame, detection_area, save_data=False)
                
        # Save negative training data
        motion_filter.no_motion_save(delta, orig_frame)

        # Display diffs
        cv.imshow('Video', veg_plot_org)
        #cv.imwrite('image/filter.jpg', veg_plot_org)
        cv.imshow('Grid Overlay', grid_frame)
        #cv.imwrite('image/grid.jpg', grid_frame)
        cv.imshow('Vegetation Filter', motion)
        #cv.imshow('Motion Filter', filtered_frame)
        else:
            print("Waiting for background to stabilize")

    if cv.waitKey(20) & 0xFF == ord('d'): # stop looping, on videos after 20 miliseconds or when "d" is pressed
        break
    elif motion_filter.downloads > detection_limit:
        break

cap.release() # closes video file
cv.destroyAllWindows() # closes all windows


