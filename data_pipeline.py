import cv2 as cv
import numpy as np




def basic_manip(frame, counter, total_frames):
    resized = cv.resize(frame, (640,640), interpolation=cv.INTER_AREA)
    txt = str(counter) + "/" + str(total_frames)
    cv.putText(resized, txt, (0,500), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0), 2)

    return resized


def pre_process(frame, learning_rate):
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    proc_image = fgbg.apply(gray_img, lrn_rate)

    return proc_image


fgbg = cv.createBackgroundSubtractorMOG2()
lrn_rate = -1
video_file = 'bird_am_vid.mp4'
counter = 0
cap = cv.VideoCapture(1)
total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
total_frames = 1

while True:
    flag, frame = cap.read()

    if not flag or frame is None or frame.size == 0:
        print("⚠️ Empty or invalid frame, skipping color conversion")
        break
    else:
        orig_frame = basic_manip(frame, counter, total_frames)
        proc_frame = basic_manip(frame, counter, total_frames)
        img_processed = pre_process(proc_frame, lrn_rate)

        cv.imshow('Video', orig_frame)
        cv.imshow('Proc_Video', img_processed)

        if cv.waitKey(20) & 0xFF == ord('d'): # stop looping on videos after 20 miliseconds or when "d" is pressed
            break

    counter = counter + 1

cap.release() # closes video file
cv.destroyAllWindows() # closes all windows


    