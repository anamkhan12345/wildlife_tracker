import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
import cv2 as cv
from datetime import datetime

def create_labels_df(dir, ext, delim):
    dir_ext = dir + "\*" + ext
    files = glob.glob(dir_ext)
    areas = []
    times = []
    detections = []
    for f in files:
        name = os.path.basename(f)  
        parts = name.split(delim)
        aArea = float(parts[2])
        areas.append(aArea)
        aTime = float(parts[0])
        times.append(aTime)
        # Get total detections
        with open(f, "r") as aFile:
            num_lines = sum(1 for _ in aFile)
            detections.append(num_lines)

    time_dt = pd.to_datetime(times, unit='ms', utc=True).tz_convert('US/Eastern')
    hours_dt = [x.hour for x in time_dt]

    df = pd.DataFrame({
        "files": files,
        "areas": areas,
        "detections": detections,
        "times": time_dt,
        "hours": hours_dt
        })
    
    df = df.sort_values("times")

    return df

def create_neg_labels(dir, ext, delim):
    
    dir_ext = dir + "\*" + ext
    files = glob.glob(dir_ext)
    areas = [0] * len(files)
    times = []
    hours = []
    detections = []
    for f in files:
        name = os.path.basename(f) 
        parts = name.split(delim)
        time_string = parts[1] + '_' + parts[2]
        aTime = datetime.strptime(time_string, "%Y%m%d_%H%M%S")
        times.append(aTime)
        aHour = aTime.hour
        hours.append(aHour)
        # Get total detections
        with open(f, "r") as aFile:
            num_lines = sum(1 for _ in aFile)
            detections.append(num_lines)


    df = pd.DataFrame({
        "files": files,
        "areas": areas,
        "detections": detections,
        "times": times,
        "hours": hours
        })
    
    df = df.sort_values("times")

    return df


def remove_files(files):
    for f in files:
        name, _ = os.path.splitext(f)
        jpg_file = name + '.jpg'
        if os.path.exists(f):
            os.remove(f)
        else:
            print(f'Could not find {f} file')

        if os.path.exists(jpg_file):
            os.remove(jpg_file)
        else:
            print(f"Could not find {jpg_file} file")


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

def yolo_format_verif(txt_file, jpg_file):
    img = cv.imread(jpg_file)
    cv.imshow('og', img)
    copy = img.copy()
    # class x_center y_center width height
    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            flt_parts = [float(x) * 640 for x in parts[1:]]
            left = ( int(flt_parts[0] - flt_parts[2]) , int(flt_parts[1] - flt_parts[3]) )
            right = ( int(flt_parts[0] + flt_parts[2]) , int(flt_parts[1] + flt_parts[3]) )
            cv.rectangle(copy, left, right, (0, 255, 0), 1)
            cv.circle(copy, 
                      (int(flt_parts[0]), int(flt_parts[1])),
                      radius=0, 
                      color=(0, 0, 255), 
                      thickness=2)
            cv.imshow('copy', copy)
    
    cv.waitKey(0)

def train_val_test_split(df):
    ''' This will split the data based on time of observations'''

    df = df.sort_values("hours")
    df = df.sort_values("hours")
    n_total = len(df)
    train_end = int(n_total * 0.7)
    val_end = int(n_total * 0.85)

    file_split = {
        'train': df['files'].iloc[:train_end].tolist(),
        'val': df['files'].iloc[train_end:val_end].tolist(),
        'test': df['files'].iloc[val_end:].tolist()
    }

    # Verification
    total_rows = sum(len(file_split[key]) for key in file_split)
    if total_rows == n_total:
        print("You got all the df rows")
    else:
        print(f'ERROR: df size: {n_total}, you got: {total_rows}')



    return file_split




# # Data verification
# txt_file = Path('image\\test\\medium\\1755723722933_area_102_3.txt')
# jpg_file = Path('image\\test\\medium\\1755723722933_area_102_3.jpg')
# yolo_format_verif(txt_file, jpg_file)


# Read in all the .jpg files
img_path = "C:\\Users\\anamk\\projects\\wildlife_tracker\\image\\test\\image"
lbl_path = "C:\\Users\\anamk\\projects\\wildlife_tracker\\image\\test\\label"
neg_lbl_path = "C:\\Users\\anamk\\projects\\wildlife_tracker\\image\\negative\\label"
df_label = create_labels_df(lbl_path, '.txt', '_')
file_split = train_val_test_split(df_label)
breakpoint()
df_neg_label = create_neg_labels(neg_lbl_path,'.txt', '_')

# Plot areas captured and detection times
# TODO: This is only the MAX area detected in each frame, not all bounding boxes
# fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# sns.histplot(data=df, x= 'areas', ax=axes[0,0])
# axes[0,0].set_title("Areas")

# sns.scatterplot(data=df, x='hours', y='areas', ax = axes[0,1])
# axes[0,1].set_title('Hours vs. Areas')

# sns.scatterplot(data=df, x='hours', y='detections', ax = axes[1,0])
# axes[1,0].set_title('Hour vs. Detections')

# sns.histplot(data=df, x='detections', ax = axes[1,1])
# axes[1,1].set_title('Detections')

# plt.show()

# Remove irrelevant
# rem_det = df[df['detections'] > 5 ]
# remove_files(rem_det['files'])

# # Areas over 10,000
# rem_areas = df[df['areas'] > 10000]
# remove_files(rem_areas['files'])

# TODO: Plot detections centroids on the actual original image

