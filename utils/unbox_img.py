import cv2
import numpy as np
from pathlib import Path
import glob
import data_gen
import math

def remove_bounding_boxes(label_path, output_path, box_thickness=2, visualize_mask=False):
    """Remove bounding boxes from annotated image using label file"""
    img_path = next(Path(label_path).parent.parent.rglob(f"{Path(label_path).stem}.jpg"), None)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    _, coordinates = data_gen.xywh_pixel(label_path, img_path)

    # Create mask for inpainting
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Draw each box on the mask
    for i in range(len(coordinates)):
        x1 = math.floor(coordinates[i]['x1'])
        y1 = math.floor(coordinates[i]['y1'])
        x2 = math.ceil(coordinates[i]['x2'])
        y2 = math.ceil(coordinates[i]['y2'])
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, box_thickness)

    # Visualize mask in bright red
    if visualize_mask:
        mask_viz = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2BGRA)
        mask_viz[mask == 255] = [0, 0, 255, 255]  # Bright red (BGR format)
        cv2.imwrite(output_path.replace('.jpg', '_mask_viz.jpg'), mask_viz)
    
    # Dilate mask to capture more box pixels
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)  # Increased iterations
    
    # Try INPAINT_NS instead (often better for lines)
    result = cv2.inpaint(img, mask, inpaintRadius=7, flags=cv2.INPAINT_NS)
    cv2.imwrite(output_path, result)
    
    return result

# Usage for single image
source_dir = r'C:\Users\anamk\projects\dataSets\sandbox_copy'
all_files = glob.glob(f"{source_dir}/**/*", recursive=True) # Search Dir
label_files = [Path(f) for f in all_files if f.endswith('.txt')] # Search list - to delete

for aLabel in label_files:

    remove_bounding_boxes(aLabel, 'bird_001_clean.jpg', box_thickness=2, visualize_mask=False)