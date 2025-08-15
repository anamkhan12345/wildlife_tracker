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
        for i in range(1, num_labels):
            area = stats[i, cv.CC_STAT_AREA]
            cent = centroids[i]
            
            if area >= min_area:
                motion_found = True
                all_box_area.append(area)
                # Draw bounding box
                x, y, w, h = stats[i, cv.CC_STAT_LEFT:cv.CC_STAT_LEFT+4]
                box_width = x + w
                box_height = y + h
                cv.rectangle(original_frame, (x, y), (box_width, box_height), (0, 255, 0), 1)

                # Save vars to later write to txt file
                all_box_centroids.append(cent)
                all_box_widths.append(box_width)
                all_box_heights.append(box_height)

        if motion_found:
            self.downloads = self.downloads + 1
            print(f"Motion detected: Area={max(all_box_area)}, Position=({x},{y}), Detection Count: {self.downloads}")

            # Calculate training params for yolo labels
            img_height = original_frame.shape[0]
            img_width = original_frame.shape[1]
            yolo_width = [w / img_width for w in all_box_widths]
            yolo_height = [h / img_height for h in all_box_heights]
            yolo_cent_x = [ c[0] / img_width for c in all_box_centroids]
            yolo_cent_y = [ c[1] / img_height for c in all_box_centroids]

            # Save frame image
            timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
            img_file = f'{self.downloads}_{timestamp}_area_{max(all_box_area)}.jpg'
            label_file = f'{self.downloads}_{timestamp}_area_{max(all_box_area)}.txt'
            img_dir = Path('image/test')
            label_dir = Path('image/test')
            img_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            img_path = os.path.join(img_dir, img_file)
            label_path = os.path.join(label_dir, label_file)

            # Save image w/bounding box and yolo label .txt file
            cv.imwrite(img_path, original_frame)
            with open(label_path, 'w') as file:
                for i in range(len(yolo_width)):
                    # class x_center y_center width height
                    file.write(f'0 {yolo_cent_x[i]} {yolo_cent_y[i]} {yolo_width[i]} {yolo_height[i]}\n')
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