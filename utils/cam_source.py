import cv2 as cv
import numpy as np


webcam_path = r'C:\Users\anamk\projects\wildlife_tracker\image\yolo_bird_data\train\images\detect_1755775421319_area_57_79.jpg'
rpi_path = r'C:\Users\anamk\projects\wildlife_tracker\image\yolo_bird_data\train\images\detect_1756986638114_area_6779_149.jpg'
missed_file =  [ r'C:\Users\anamk\projects\wildlife_tracker\image\yolo_bird_data\train\images\detect_1756996041312_area_391_495.jpg', 
r'C:\Users\anamk\projects\wildlife_tracker\image\yolo_bird_data\train\images\negative_1757082123146_area_0_9.jpg',
r'C:\Users\anamk\projects\wildlife_tracker\image\yolo_bird_data\train\images\negative_1757082303154_area_0_10.jpg',
r'C:\Users\anamk\projects\wildlife_tracker\image\yolo_bird_data\train\images\negative_1757082483168_area_0_11.jpg',
r'C:\Users\anamk\projects\wildlife_tracker\image\yolo_bird_data\train\images\negative_1757082663201_area_0_12.jpg',
r'C:\Users\anamk\projects\wildlife_tracker\image\yolo_bird_data\train\images\negative_1757082843230_area_0_13.jpg',
r'C:\Users\anamk\projects\wildlife_tracker\image\yolo_bird_data\train\images\negative_1757083023293_area_0_14.jpg',
r'C:\Users\anamk\projects\wildlife_tracker\image\yolo_bird_data\train\images\negative_1757083203306_area_0_15.jpg',
r'C:\Users\anamk\projects\wildlife_tracker\image\yolo_bird_data\train\images\negative_1757083563334_area_0_17.jpg',
]

files = [webcam_path, rpi_path]

for file in missed_file:
    img = cv.imread(file)
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    edges = cv.Canny(gray, 100, 200)
    # cv.imshow('canny', edges)
    # cv.imshow('img', img)
    # cv.imshow('gray', gray)
    num_edges = np.count_nonzero(edges)
    print('Num edges:', num_edges)
    cv.waitKey(0)
