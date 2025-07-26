import cv2
import ncnn
import numpy as np
import sys
sys.path.append('..')
import detect



img_path = 'test_bird.jpg'
model_path = '../models/best.pt'
image = cv2.imread(img_path)

# Run detection
breakpoint()
detector = detect.ncnn_model(model_path)
rgb_image = detect.pre_process(image)
detection_result = detector(rgb_image)
detect.parse_detection(detection_result)

# Show result (comment out if running headless)
#cv2.imshow("Detection Result", result_image)
print("Press any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()
