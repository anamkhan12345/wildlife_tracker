import cv2 as cv
import numpy as np
import os
from pathlib import Path
import time
from collections import deque


class VegetationFilter:
    def __init__(self):
        self.backSub = cv.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,  # Higher threshold for vegetation areas
            detectShadows=True
        )
        self.vegetation_mask = None
        self.base_learning_rate = 0.01
        self.vegetation_learning_rate = 0.001  # Much slower for vegetation
        
    def set_vegetation_zones(self, frame_shape, vegetation_areas):
        h, w = frame_shape[:2]
        self.vegetation_mask = np.zeros((h, w), dtype=np.uint8)
        
        for area in vegetation_areas:
            if len(area) == 4:  # Rectangle (x1, y1, x2, y2)
                x1, y1, x2, y2 = area
                self.vegetation_mask[y1:y2, x1:x2] = 255
            else:  # Polygon
                cv.fillPoly(self.vegetation_mask, [np.array(area)], 255)
    
    def adaptive_learning(self, frame):
        # Get base motion detection
        motion_mask = self.backSub.apply(frame, learningRate=self.base_learning_rate)
        
        if self.vegetation_mask is not None:
            # Get vegetation-specific motion with slower learning
            motion_mask = cv.bitwise_and(motion_mask, cv.bitwise_not(self.vegetation_mask))
            # Combine: use vegetation_motion in vegetation areas, regular motion elsewhere
            # motion_mask = np.where(self.vegetation_mask > 0, vegetation_motion, motion_mask)

        # Basic cleanup
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        motion_mask = cv.morphologyEx(motion_mask, cv.MORPH_OPEN, kernel)

        return motion_mask


class MultipleFrameFilter:
    def __init__(self, buffer_size=5, threshold=0.6):
        self.buffer_size = buffer_size
        self.threshold = threshold  # Fraction of frames that must show motion
        self.frame_buffer = deque(maxlen=buffer_size)
        self.downloads = 0

        self.small_img_dir = Path('image/test/small')
        self.med_img_dir = Path('image/test/medium')
        self.big_img_dir = Path('image/test/big')
        self.small_img_dir.mkdir(parents=True, exist_ok=True)
        self.med_img_dir.mkdir(parents=True, exist_ok=True)
        self.big_img_dir.mkdir(parents=True, exist_ok=True)
        
    def filter_motion(self, motion_mask):
        # Add current frame to buffer
        self.frame_buffer.append(motion_mask.astype(np.float32) / 255.0)

        if len(self.frame_buffer) < self.buffer_size:
            return np.zeros_like(motion_mask)  # Not enough frames yet
        
        # Sum across time dimension
        temporal_sum = np.sum(self.frame_buffer, axis=0)
        
        # Keep pixels that show motion in enough frames
        persistent_motion = (temporal_sum >= (self.threshold * self.buffer_size))
        
        return (persistent_motion * 255).astype(np.uint8)
    
    def save_data(self, all_box_area, yolo_format, dir_path, original_frame):
        # Save frame image
        timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
        img_file = f'{timestamp}_area_{all_box_area}_{self.downloads}.jpg'
        label_file = f'{timestamp}_area_{all_box_area}_{self.downloads}.txt'

        img_path = os.path.join(dir_path, img_file)
        label_path = os.path.join(dir_path, label_file)

        # Save image w/bounding box and yolo label .txt file
        cv.imwrite(img_path, original_frame)
        with open(label_path, 'w') as file:
            print("writting")
            breakpoint()
            for i in range(len(yolo_format['width'])):
                # class x_center y_center width height
                file.write(f"0 {yolo_format['x_cent'][i]} {yolo_format['y_cent'][i]} {yolo_format['width'][i]} {yolo_format['height'][i]}\n")


    def annotate_data(self, persistent_motion, original_frame, min_area):

        # Find connected components
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
            persistent_motion, connectivity=8
        )
        motion_found = False
        all_box_widths = []
        all_box_heights = []
        all_box_centroids = []
        all_box_area = []

        # Skip background label (0)
        if num_labels < 6:
            # Loop through all groups and draw bounding box
            for i in range(1, num_labels):
                area = stats[i, cv.CC_STAT_AREA]
                cent = centroids[i]
                if area >= min_area and area < 7500:
                    motion_found = True
                    all_box_area.append(area)
                    # Draw bounding box
                    x, y, w, h = stats[i, cv.CC_STAT_LEFT:cv.CC_STAT_LEFT+4]
                    bttm_x = x + w
                    bttm_y = y + h
                    cv.rectangle(original_frame, (x, y), (bttm_x, bttm_y), (0, 255, 0), 1)
                    # Save vars to later write to txt file
                    all_box_centroids.append(cent)
                    all_box_widths.append(w)
                    all_box_heights.append(h)

            # After drawing all bounding boxes - save img with drawings and yolo formatted text file
            if motion_found:
                self.downloads = self.downloads + 1
                max_area = max(all_box_area)
                print(f"Motion detected: Area={max_area}, Position=({x},{y}), Detection Count: {self.downloads}")

                # Calculate training params for yolo labels
                img_height = original_frame.shape[0]
                img_width = original_frame.shape[1]
                yolo_width = [w / img_width for w in all_box_widths]
                yolo_height = [h / img_height for h in all_box_heights]
                yolo_cent_x = [ round(c[0] / img_width, 6) for c in all_box_centroids]
                yolo_cent_y = [ round(c[1] / img_height, 6) for c in all_box_centroids]
                yolo_format = {'x_cent': yolo_cent_x,
                               'y_cent': yolo_cent_y,
                               'width': yolo_width,
                               'height': yolo_height}
                breakpoint()

                if max_area < 50:
                    self.save_data(max_area, yolo_format, self.small_img_dir, original_frame)
                elif max_area >= 50 and max_area < 500:
                    self.save_data(max_area, yolo_format, self.med_img_dir, original_frame)
                else:
                    self.save_data(max_area, yolo_format, self.big_img_dir, original_frame)
        else:
            print("Too much motion detected, camera possibly moving")

        return num_labels - 1  # Return number of motion groups found


def add_grid(image, rows=3, cols=3, color=(255, 255, 255), thickness=2, alpha=0.8):
    result = image.copy()
    h, w = image.shape[:2]
    overlay = image.copy()
    
    # Draw vertical lines
    for i in range(1, cols):
        x = int(w * i / cols)
        cv.line(overlay, (x, 0), (x, h), color, thickness)
    
    # Draw horizontal lines  
    for i in range(1, rows):
        y = int(h * i / rows)
        cv.line(overlay, (0, y), (w, y), color, thickness)
    
    # Blend overlay with original image
    result = cv.addWeighted(result, 1-alpha, overlay, alpha, 0)
    
    return result

def plot_zone(frame, veg_zone):
        img = frame.copy()
        for area in veg_zone:
            if len(area) == 4:  # Rectangle (x1, y1, x2, y2)
                x1, y1, x2, y2 = area
                zone_plot = cv.rectangle(img, (x1,y1), (x2,y2), (0,255,0), thickness=2)
            else:  # Polygon
                zone_plot = cv.polylines(img, [np.array(area)], isClosed=True, color=(0, 255, 0), thickness=3)

        return zone_plot