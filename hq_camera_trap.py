from picamera2 import Picamera2
import cv2 as cv
import motion_class
import bird_model
import argparse
import time


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

    while True:
        frame = picam.capture_array()
        counter = counter + 1
        if counter > 30:
            # Show the re-sized webcam images
            orig_frame = frame
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
                detector.set_vegetation_zones(orig_frame.shape, veg_zone)

                # Find motion
                motion = detector.adaptive_learning(gray)
                motion_blur = detector.adaptive_learning(blur)

                # Filter motion found
                detection = motion_filter.motion_filter(motion, orig_frame, detection_area, save_data=False)

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
    parser.add_argument('-l', '--det_cnt_limit', type=int, default=50)

    # Parse arguments
    args = parser.parse_args()
    cam_id = args.cam
    min_area = args.det_area
    detection_limit = args.det_cnt_limit

    # Run pipeline
    hq_cam_trap(cam_id, min_area, detection_limit)

if __name__ == '__main__':
    main()

