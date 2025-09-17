import os
import glob
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
import cv2 as cv
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS

def check_image_label_matches(source_dir):

    all_files = glob.glob(f"{source_dir}/**/*", recursive=True)
    image_files = [Path(f) for f in all_files if f.endswith('.jpg')]
    label_files = [Path(f) for f in all_files if f.endswith('.txt')]

    label_stems = {f.stem for f in label_files}
    image_stems = {f.stem for f in image_files}
    missing_labels = []
    missing_image = []

    for img_file in image_files:
        if img_file.stem not in label_stems:
            missing_labels.append(img_file)
            print(f"✗ Missing label: {img_file.stem}.txt for {img_file}")

    for label_file in label_files:
        if label_file.stem not in image_stems:
            missing_image.append(label_file)
            print(f"✗ Missing image: {label_file.stem}.jpg for {label_file}")

    if not missing_labels and not missing_image:
        print(f"✓ Perfect! All {len(image_files)} images have matching labels")
        img_file = image_files[0]
        txt_file = next(img_file.parent.parent.rglob(f"{img_file.stem}.txt"), None)
        yolo_format_verif(txt_file, img_file)
    else:
        print(f"✗ Found {len(missing_labels)} images without matching labels")

    return (len(missing_labels) + len(missing_image) == 0), missing_labels, missing_image

def extract_meta_data(jpg_file):
 with Image.open(jpg_file) as img:
        # Basic image info
        info = {
            'filename': jpg_file,
            'format': img.format,
            'mode': img.mode,
            'size': img.size,
            'width': img.width,
            'height': img.height
        }
        
        # EXIF data if available
        exifdata = img.getexif()
        if exifdata:
            for tag_id, value in exifdata.items():
                tag = TAGS.get(tag_id, tag_id)
                info[tag] = value
                
        return info

def create_df(dir, delim):
    dir_ext = dir + "\**\*.txt"
    files = glob.glob(dir_ext, recursive=True)
    max_areas = []
    times = []
    detections = []
    jpg_files = []
    jpg_size = []
    box_area = []
    box_cent = []
    x_1 = []
    y_1 = []
    x_2 = []
    y_2 = []
    edges = []
    source = []
    ar = []
    data_class = []

    for f in files:
        name = os.path.basename(f)
        parts = name.split(delim)
        if parts[0] == 'detect':
            aClass = 'bird'
        else:
            aClass = 'negative'
        data_class.append(aClass)

        aArea = float(parts[3])
        max_areas.append(aArea)
        aTime = float(parts[1])
        times.append(aTime)

        # Find corresponding jpg file
        jpg = next(Path(f).parent.parent.rglob(f"{Path(f).stem}.jpg"), None)
        jpg_files.append(str(jpg))
        if jpg is None:
            print(f"**** Could not find jpg for {f} *****")
            exit(1)
        else:
            # Extract bbox info
            img_size, coordinates = xywh_pixel(f, jpg)
            img_size_str = str(img_size)
            aspect_ratio = ( img_size[0] / img_size[1] )
            bbox_areas = [x['width']* x['height'] for x in coordinates]
            bbox_centroids = [(x['x_cent'], x['y_cent']) for x in coordinates]
            x1 = [x['x1'] for x in coordinates]
            y1 = [x['y1'] for x in coordinates]
            x2 = [x['x2'] for x in coordinates]
            y2 = [x['y2'] for x in coordinates]

            # Set up for df
            jpg_size.append(img_size_str)
            ar.append(aspect_ratio)
            box_area.append(bbox_areas)
            box_cent.append(bbox_centroids)
            x_1.append(x1)
            y_1.append(y1)
            x_2.append(x2)
            y_2.append(y2)

        # Get total detections
        with open(f, "r") as aFile:
            num_lines = sum(1 for _ in aFile)
            detections.append(num_lines)

    time_dt = pd.to_datetime(times, unit='ms', utc=True).tz_convert('US/Eastern')
    hours_dt = [x.hour for x in time_dt]
    df = pd.DataFrame({
        "class": data_class,
        "detections": detections,
        "files": files,
        "jpg_files": jpg_files,
        "jpg_size": jpg_size,
        "aspect_ratio": ar,
        "max_areas": max_areas,
        "bbox_area": box_area,
        "bbox_centroid": box_cent,
        "bbox_x1":x_1,
        "bbox_y1":y_1,
        "bbox_x2":x_2,
        "bbox_y2":y_2,
        "times": time_dt,
        "hours": hours_dt
        })
    
    df = df.sort_values("times")

    return df


def remove_files(files):
    for f in files:
        if os.path.exists(f):
            os.remove(f)
        else:
            print(f'Could not find {f} file')

def re_order(df):
    small_dir = Path('image/small')
    med_dir = Path('image/medium')
    big_dir = Path('image/big')

    small_dir.mkdir(parents=True, exist_ok=True)
    med_dir.mkdir(parents=True, exist_ok=True)
    big_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        head, tail = os.path.split(row['files'])
        tail_no_ext, _ = os.path.splitext(tail)
        tail_jpg = tail_no_ext + '.jpg'
        tail_jpg_full = os.path.join(head, tail_jpg)

        if os.path.exists(row['files']):
            if row['areas'] < 50:
                shutil.move(row['files'], os.path.join(small_dir, tail))
                shutil.move(tail_jpg_full, os.path.join(small_dir, tail_jpg))
            elif row['areas'] > 50 and row['areas'] < 500:
                shutil.move(row['files'], os.path.join(med_dir, tail))
                shutil.move(tail_jpg_full, os.path.join(med_dir, tail_jpg))
            else:
                shutil.move(row['files'], os.path.join(big_dir, tail))
                shutil.move(tail_jpg_full, os.path.join(big_dir, tail_jpg))
        else:
            print(f'**** Could not find {tail} *****')

def xywh_pixel(txt_file, jpg_file):
    img = cv.imread(jpg_file)
    width = int(img.shape[1])
    height = int(img.shape[0])
    img_w_h = [width, height]
    coordinates = []

    # class x_center y_center width height
    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            flt_parts = [float(x) for x in parts[1:]]

            # Find pixel values from yolo formatting
            x_cent = float(flt_parts[0]) * width
            y_cent = float(flt_parts[1]) * height
            width_pix = float(flt_parts[2]) * width
            height_pix = float(flt_parts[3]) * height

            # Find boudning box pixel values from yolo formatting
            x1 = x_cent - (width_pix / 2)
            y1 = y_cent - (height_pix / 2)
            x2 = x_cent + (width_pix / 2)
            y2 = y_cent + (height_pix / 2)

            bbox_info = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 
                         'x_cent': x_cent, 'y_cent': y_cent,
                         'width': width_pix, 'height': height_pix}

            coordinates.append(bbox_info)


    return img_w_h, coordinates

def yolo_format_verif(txt_file, jpg_file, show_comp=False):
    img  = cv.imread(jpg_file)
    copy = img.copy()
    _, coordinates = xywh_pixel(txt_file, jpg_file)
    for i in range(len(coordinates)):
        x1 = coordinates[i]['x1']
        y1 = coordinates[i]['y1']
        x2 = coordinates[i]['x2']
        y2 = coordinates[i]['y2']
        x_cent = coordinates[i]['x_cent']
        y_cent = coordinates[i]['y_cent']

        cv.rectangle(copy, 
                    (int(x1), int(y1)), 
                    (int(x2), int(y2)), 
                    color=(0, 255, 0), 
                    thickness=2)

        cv.circle(copy, 
                    (int(x_cent), int(y_cent)),
                    radius=0, 
                    color=(0, 0, 255), 
                    thickness=2)


    cv.imshow('copy', copy)
    cv.imshow('og', img)
    cv.waitKey(0)

    return coordinates


def train_val_test_split(df):
    ''' This will split the data based on time of observations'''

    df = df.sort_values("hours")
    df = df.sort_values("hours")
    n_total = len(df)
    train_end = int(n_total * 0.7)
    val_end = int(n_total * 0.85)

    file_split = {
        'train': df['files'].iloc[:train_end].tolist() + df['jpg_files'].iloc[:train_end].tolist(),
        'val': df['files'].iloc[train_end:val_end].tolist() + df['jpg_files'].iloc[train_end:val_end].tolist(),
        'test': df['files'].iloc[val_end:].tolist() + df['jpg_files'].iloc[val_end:].tolist()
    }
    # Verification
    total_rows = sum(len(file_split[key]) for key in file_split) / 2
    if total_rows == n_total:
        print("You got all the df rows")
    else:
        print(f'ERROR: df size: {n_total}, you got: {total_rows}')

    return file_split


def copy_files_to_yolo_structure(file_split, output_dir):
    base_path = Path(output_dir)
    for split in ['train', 'val', 'test']:
        img_dir = base_path / split / 'images'
        label_dir = base_path / split / 'labels'

        # Create YOLO directory structure
        (img_dir).mkdir(parents=True, exist_ok=True)
        (label_dir).mkdir(parents=True, exist_ok=True)

        for filename in file_split[split]:
            file_path = Path(filename)
            # Destination paths
            if file_path.suffix == '.txt':
                dest_path = label_dir / file_path.name
                shutil.move(str(file_path), str(dest_path))
            elif file_path.suffix == '.jpg':
                dest_path = img_dir / file_path.name
                shutil.move(str(file_path), str(dest_path))

    print("YOLO structure creation complete!")

# # Data verification
# txt_file = Path('image\\test\\medium\\1755723722933_area_102_3.txt')
# jpg_file = Path('image\\test\\medium\\1755723722933_area_102_3.jpg')
# yolo_format_verif(txt_file, jpg_file)


# Read in all the label files
def check():
    input_dir = r"C:\Users\anamk\projects\wildlife_tracker\image\yolo_bird_data"

    # Verify that each label file has corresponding .jpg file
    yolo_formatted, missing_labels, missing_images = check_image_label_matches(input_dir)
    print(f"Yolo formatted: {yolo_formatted}")
    if not yolo_formatted:
        exit()

    # Create dataframe with detection info
    df = create_df(input_dir, delim='_')

    # # # Test train val split for detections and negative detectoins
    # file_split_1 = train_val_test_split(df)
    # copy_files_to_yolo_structure(file_split_1, output_dir)


if __name__ == "__main__":
    check()