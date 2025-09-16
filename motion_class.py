import cv2 as cv
import numpy as np
import os
from pathlib import Path
import time
from collections import deque
from datetime import datetime


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
        self.neg_counter = 0
        self.last_neg = 0
        self.debug = False 
        # params to reset 
        self.motion_found = False
        self.all_box_widths = []
        self.all_box_heights = []
        self.all_box_centroids = []
        self.all_box_area = []
        self.frame_detect = 0

        self.small_img_dir = Path('image/test/small')
        self.med_img_dir = Path('image/test/medium')
        self.big_img_dir = Path('image/test/big')
        self.neg_img_dir = Path('image/negative')
        self.small_img_dir.mkdir(parents=True, exist_ok=True)
        self.med_img_dir.mkdir(parents=True, exist_ok=True)
        self.big_img_dir.mkdir(parents=True, exist_ok=True)
        self.neg_img_dir.mkdir(parents=True, exist_ok=True)

    def reset_vars(self):
        self.motion_found = False
        self.all_box_widths = []
        self.all_box_heights = []
        self.all_box_centroids = []
        self.all_box_area = []
        self.frame_detect = 0

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


    # def intensity_filter(self, original_frame, label, stats, brt_thresh=200):
    #     if len(original_frame) == 3:
    #         gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    #     else:
    #         original_frame
    #     valid_detects = []
        
    #     for i in range(1, num_labels):
    #         possible_det = (labels == i)
    #         avg_brightness = np.mean(gray[component_mask])
    #     if avg_brightness <  brt_thresh:
    #         valid_detects.append(i) 

    #     return valid_components

    def save_data(self, max_area, yolo_format, dir_path, bbox_frame, og_frame):
        # Save frame image
        timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
        img_file = f'detect_{timestamp}_area_{max_area}_{self.downloads}.jpg'
        raw_file = f'raw_{timestamp}_area_{max_area}_motion_{self.downloads}.jpg'
        label_file = f'detect_{timestamp}_area_{max_area}_{self.downloads}.txt'
        
        img_path = os.path.join(dir_path, img_file)
        raw_path = os.path.join(dir_path, raw_file)
        label_path = os.path.join(dir_path, label_file)

        # Save image w/bounding box and yolo label .txt file
        cv.imwrite(img_path, bbox_frame)
        cv.imwrite(raw_path, og_frame)
        with open(label_path, 'w') as file:
            for i in range(len(yolo_format['width'])):
                # class x_center y_center width height
                file.write(f"0 {yolo_format['x_cent'][i]} {yolo_format['y_cent'][i]} {yolo_format['width'][i]} {yolo_format['height'][i]}\n")

    def yolo_annotation(self, og_frame, save_data):
        # After drawing all bounding boxes - save img with drawings and yolo formatted text file
        if self.motion_found:
            frame = og_frame.copy()
            self.downloads = self.downloads + 1
            max_area = max(self.all_box_area)
            print(f"Motion detected: Area={max_area}, Detection Count: {self.downloads}")

            # Calculate training params for yolo labels
            img_height = frame.shape[0]
            img_width = frame.shape[1]
            yolo_width =  [round(w / img_width, 6) for w in self.all_box_widths]
            yolo_height = [round(h / img_height, 6) for h in self.all_box_heights]
            yolo_cent_x = [ round(c[0] / img_width, 6) for c in self.all_box_centroids]
            yolo_cent_y = [ round(c[1] / img_height, 6) for c in self.all_box_centroids]
            yolo_format = {'x_cent': yolo_cent_x,
                           'y_cent': yolo_cent_y,
                           'width': yolo_width,
                           'height': yolo_height}
            if save_data:
                if max_area < 50:
                    self.save_data(max_area, yolo_format, self.small_img_dir, frame, og_frame)
                elif max_area >= 50 and max_area < 500:
                    self.save_data(max_area, yolo_format, self.med_img_dir, frame, og_frame)
                else:
                    self.save_data(max_area, yolo_format, self.big_img_dir, frame, og_frame)

    def motion_filter(self, persistent_motion, original_frame, min_area, save_data=True):

        # Reset class vars
        self.reset_vars()
        # Find connected components
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
            persistent_motion, connectivity=8
        )
	# TODO: Filter out clouds based on intensity
        #valid_detects = intensity_filter(gray_frame, label, stats, brt_thresh=200)

        timestamp = int(time.time() * 1000) 
        # Skip background label (0)
        if (num_labels < 6 and num_labels > 1) or self.debug:
            # Loop through all groups and draw bounding box
            for i in range(1, num_labels):
                area = stats[i, cv.CC_STAT_AREA]
                if area >= min_area and area < 1036800:
                    self.motion_found = True
                    self.frame_detect = len(num_labels)
                    self.all_box_area.append(area)
                    # Draw bounding box
                    x, y, w, h = stats[i, cv.CC_STAT_LEFT:cv.CC_STAT_LEFT+4]
                    bttm_x = x + w
                    bttm_y = y + h
                    cv.rectangle(original_frame, (x, y), (bttm_x, bttm_y), (0, 255, 0), 2)
                    # Calculate bounding box center
                    center_x = x + w / 2.0
                    center_y = y + h / 2.0
                    centroid = (center_x, center_y)
                    # Save vars to later write to txt file
                    self.all_box_centroids.append(centroid)
                    self.all_box_widths.append(w)
                    self.all_box_heights.append(h)
                    # Generate YOLO formatting docs
                    self.yolo_annotation(original_frame, save_data)
        else:
            print(f"Too much motion detected: {timestamp}")

        return num_labels - 1  # Return number of motion groups found


    def no_motion_save(self, delta, frame):
        # Check if it's time to capture
        current_time = time.time()
        if current_time - self.last_neg >= delta and not self.motion_found:
            self.neg_counter = self.neg_counter + 1
            # Save image
            timestamp = int(time.time() * 1000)
            filename = f"negative_{timestamp}_area_0_{self.neg_counter}.jpg"
            filepath = os.path.join(self.neg_img_dir, filename) 
            cv.imwrite(filepath, frame)

            # Create empty text file for negative image
            neg_txt_file = f"negative_{timestamp}_area_0_{self.neg_counter}.txt"
            neg_file_path = os.path.join(self.neg_img_dir, neg_txt_file)
            Path(neg_file_path).touch()

            # Create empty text file to go with image
            self.last_neg = current_time
            print(f"saved random image: {self.neg_counter}")


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
