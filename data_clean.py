import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.units as munits

#munits.registry.clear()

def create_data_df(dir, ext, delim):
    dir_ext = dir + "\*" + ext
    files = glob.glob(dir_ext)
    areas = []
    times = []

    for f in files:
        name, _ = os.path.splitext(f)
        parts = name.split(delim)
        aArea = float(parts[-1])
        areas.append(aArea)
        aTime = float(parts[1])
        times.append(aTime)

    time_dt = pd.to_datetime(times, unit='ms', utc=True).tz_convert('US/Eastern')
    hours_dt = [x.hour for x in time_dt]

    df = pd.DataFrame({
        "files": files,
        "areas": areas,
        "times": time_dt,
        "hours": hours_dt
        })
    
    df = df.sort_values("times")

    return df


# Read in all the .jpg files
df_test = create_data_df('image\\test', '.jpg', '_')
df_test2 = create_data_df('image\\test2', '.jpg', '_')
df_train = create_data_df('image\\train', '.jpg', '_')
df = pd.concat([df_test, df_test2, df_train], ignore_index=True)
df = df.sort_values('times')
breakpoint()

# Plot areas captured and detection times
# TODO: This is only the MAX area detected in each frame, not all bounding boxes
sns.scatterplot(df['areas'])
plt.show()
sns.scatterplot(data=df, x='hours', y='areas')
plt.show()

# Explore data further

# Remove irrelevant areas 


# Read in the .txt files 

# For each .txt file list number of detections based on line number

# Plot the amount of detections in a frame 

# Remove outlier detections - those where camera is moving

# Plot detections centroids on the actual original image

