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
rgb_image = detect.pre_process(image)
detector = detect.ncnn_model(model_path)
detection_result = detector(rgb_image)
breakpoint()
detect.parse_detection(detection_result)


#cv2.imshow("Detection Result", result_image)
print("Press any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()
