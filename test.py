import bird_model
import cv2 as cv


detector = bird_model.BirdModel()
model_path = r'models/best_09082025.pt'
detector.ncnn_model(model_path)

img_path = r'C:\Users\anamk\projects\wildlife_tracker\image\yolo_bird_data\train\images\detect_1756986919353_area_454_288.jpg'
img = cv.imread(img_path)
result = detector.model(img)
detector.parse_detection(result, img)