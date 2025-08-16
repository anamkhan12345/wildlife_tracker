import os
import glob
import pandas as pd

def create_data_df(dir, ext, delim):
    dir_ext = dir + "\*" + ext
    files = glob.glob(dir_ext)
    areas = []
    times = []

    for f in files:
        name, _ = os.path.splitext(f)
        parts = name.split(delim)
        areas.append(parts[-1])
        times.append(parts[1])
    
    df = pd.DataFrame({
        "files": files,
        "area": areas,
        "time": times
        })

    return df


# Read in all the .jpg files
df_test = create_data_df('image\\test', '.jpg', '_')
breakpoint()
# Parse out the timestamp from file name

# Plot areas captured

# Plot detection times

# Remove irrelevant areas 


# Read in the .txt files 

# For each .txt file list number of detections based on line number

# Plot the amount of detections in a frame 

# Remove outlier detections - those where camera is moving

# Plot detections centroids on the actual original image

