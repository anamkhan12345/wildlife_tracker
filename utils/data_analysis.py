import data_clean
import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
import cv2 as cv


def plot_set_1(df):
    # Plot areas captured and detection times
    #TODO: This is only the MAX area detected in each frame, not all bounding boxes
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    sns.histplot(data=df, x= 'areas', ax=axes[0,0])
    axes[0,0].set_title("Areas")

    sns.scatterplot(data=df, x='hours', y='areas', ax = axes[0,1])
    axes[0,1].set_title('Hours vs. Areas')

    sns.scatterplot(data=df, x='hours', y='detections', ax = axes[1,0])
    axes[1,0].set_title('Hour vs. Detections')

    sns.histplot(data=df, x='detections', ax = axes[1,1])
    axes[1,1].set_title('Detections')

    plt.show()

def plot_set_2(df):
    # Plot centroid locations
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))

    # Flatten all tuples from all rows
    all_tuples = [tuple for tuple_list in df['bbox_centroid'] for tuple in tuple_list]

    # Extract x,y coordinates
    x_vals = [t[0] for t in all_tuples]
    y_vals = [t[1] for t in all_tuples]

    sns.scatterplot(x=x_vals, y=y_vals, ax=axes[0,0])
    axes[0,0].set_xlabel('X coordinate')
    axes[0,0].set_ylabel('Y coordinate')
    axes[0,0].set_title("Centroids")

    # Plot all areas
    all_areas = [area for area_list in df['bbox_area'] for area in area_list ]
    sns.histplot(data=all_areas, ax=axes[0,1])
    axes[0,1].set_title('Bounding Box Areas')

    # Plot image sizes
    sns.countplot(data=df, x='jpg_size', ax=axes[0,2])
    axes[0,2].set_title('Image Sizes')
    
    # Count of Aspect Ratios
    sns.countplot(data=df, x='aspect_ratio', ax=axes[1,0])
    axes[1,0].set_title('Aspect Ratios')

    # Count of classes
    sns.countplot(data=df, x='class', ax=axes[1,1])
    axes[1,1].set_title('Class Counts')

    plt.show()



input_dir = r'C:\Users\anamk\projects\wildlife_tracker\image\yolo_bird_data'

# Create dataframe with detection info
df = data_clean.create_df(input_dir, delim='_')
plot_set_2(df)
