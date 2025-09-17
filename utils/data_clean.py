import cv2 as cv
import numpy as np
import pandas as pd
import data_gen as data_gen
from data_analysis import df_lum


# Create dataframe from image directory
input_dir = r'C:\Users\anamk\projects\wildlife_tracker\image\yolo_bird_data'
df_img = data_gen.create_df(input_dir, delim='_')

# Ignore files with multiple detections and a detection area greater than 100 pixels
df = df_img[(df_img['detections'] == 1) & (df_img['max_areas'] < 100)]

# Brightness and contrast analysis
df_lum_set = df_lum(df['jpg_files'])
# Remove underexposed images
underexposed_files = df_lum_set[df_lum_set['underexposed_ratio'] > 0.3]['filename']
breakpoint()
data_gen.remove_files(underexposed_files)

