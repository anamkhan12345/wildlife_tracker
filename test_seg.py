from ncnn_det import NCNNDetector
import cv2
import ncnn
import numpy as np


detector = NCNNDetector('yolo11n_ncnn_model/model.ncnn.param', 'yolo1n_ncnn_model/model.ncnn.bin')
img_path = 'img_20250720_213631.jpg'

img = cv2.imread(img_path)

# Run detection
breakpoint()
output = detector.detect(img)
detections = detector.postprocess(output, img.shape)

# Visualize
result = draw_detections(image, detections, 'who')

# Show result (comment out if running headless)
cv2.imshow("Detection Result", result_image)
print("Press any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()
