import cv2 as cv
import pipeline_class

video_file = 'bird_am_vid.mp4'
counter = 0
cap = cv.VideoCapture(1)
total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
total_frames = 1

# Define vegetation areas as rectangles (x1, y1, x2, y2)
detector = pipeline_class.VegetationFilter()
veg_zone = [
    (512, 256, 639, 639),      # Tree on left
]

# Motion filter over frames
motion_filter = pipeline_class.MultipleFrameFilter(buffer_size=5, threshold=0.6)

while True:
    flag, frame = cap.read()

    if frame is None or frame.size == 0:
        print("Empty or invalid frame, skipping color conversion")
    elif not flag:
        break
    else:
        # Show the re-sized webcam images
        orig_frame = cv.resize(frame, (640,640), interpolation=cv.INTER_AREA)
        txt = str(counter) + "/" + str(total_frames)
        cv.putText(orig_frame, txt, (0,500), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0), 2)

        # Grid overlay
        grid_frame = pipeline_class.add_grid(orig_frame, rows=10, cols=10, thickness=1, alpha=0.5)
        
        # Check veg zone
        x1, y1, x2, y2 = veg_zone[0]
        veg_Zone = cv.rectangle(orig_frame, (x1,y1), (x2,y2), (0,255,0), thickness=2) # frame, pt1, pt2, color, thickness (-1 fills it in)

        # Set Vegetation areas
        detector.set_vegetation_zones(orig_frame.shape, veg_zone)

        # Find motion mask - remove swaying vegetation
        motion = detector.adaptive_learning(orig_frame)

        # Filter for motion across multiple frames
        filtered_frame = motion_filter.filter_motion(motion)

        # Display diffs
        cv.imshow('Video', orig_frame)
        #cv.imshow('Grid Overlay', grid_frame)
        cv.imshow('Veg Zone', veg_Zone)
        cv.imshow('Vegetation Filter', motion)
        cv.imshow('Motion Filter', filtered_frame)



    if cv.waitKey(20) & 0xFF == ord('d'): # stop looping on videos after 20 miliseconds or when "d" is pressed
        break

    counter = counter + 1

cap.release() # closes video file
cv.destroyAllWindows() # closes all windows


    