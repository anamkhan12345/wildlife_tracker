import cv2 as cv
import pipeline_class


run_type = 'test'

if run_type is 'test':
    video_file = 'bird_am_vid.mp4'
    cap = cv.VideoCapture(video_file)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
else:
    cap = cv.VideoCapture(1)
    total_frames = 1

counter = 0

# Define vegetation areas as rectangles (x1, y1, x2, y2)
detector = pipeline_class.VegetationFilter()
veg_zone = [
    (448, 192, 639, 639),
    (512, 0, 639, 192)

]

# Motion filter over frames
motion_filter = pipeline_class.MultipleFrameFilter(buffer_size=5, threshold=0.6)

while True:
    flag, frame = cap.read()

    if not flag or frame is None or frame.size == 0:
        print("Empty or invalid frame, skipping color conversion")
        break
    else:
        # Show the re-sized webcam images
        orig_frame = cv.resize(frame, (640,640), interpolation=cv.INTER_AREA)

        # Grid overlay
        grid_frame = pipeline_class.add_grid(orig_frame, rows=10, cols=10, thickness=1, alpha=0.5)

        # Check veg zone
        veg_plot_org = pipeline_class.plot_zone(orig_frame, veg_zone)
        txt = str(counter) + "/" + str(total_frames)
        cv.putText(veg_plot_org, txt, (0,639), 
                   cv.FONT_HERSHEY_TRIPLEX, 0.5,
                   (0,255,0), 1, lineType=cv.LINE_AA)

        # Set Vegetation areas
        detector.set_vegetation_zones(orig_frame.shape, veg_zone)

        # Find motion mask - remove swaying vegetation
        motion = detector.adaptive_learning(orig_frame)

        # Filter for motion across multiple frames
        filtered_frame = motion_filter.filter_motion(motion)

        # Save any groups found
        detection = motion_filter.analyze_motion(filtered_frame, orig_frame, 100)

        # Display diffs
        cv.imshow('Video', veg_plot_org)
        #cv.imshow('Grid Overlay', grid_frame)
        cv.imshow('Vegetation Filter', motion)
        cv.imshow('Motion Filter', filtered_frame)

    if cv.waitKey(20) & 0xFF == ord('d'): # stop looping on videos after 20 miliseconds or when "d" is pressed
        break

    counter = counter + 1

cap.release() # closes video file
cv.destroyAllWindows() # closes all windows


    