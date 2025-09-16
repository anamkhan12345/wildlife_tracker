import data_clean
import seaborn as sns
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
import numpy as np

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


    # Plot all areas
    all_areas = [area for area_list in df['bbox_area'] for area in area_list ]
    sns.histplot(data=all_areas, ax=axes[0,1], kde=True)
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

def plot_coords(df):
    # Plot centroid locations
    fig, axes = plt.subplots(1, 3, figsize=(10, 8))

    # Flatten all tuples from all rows
    all_tuples = [tuple for tuple_list in df['bbox_centroid'] for tuple in tuple_list]

    # Extract centroid x,y coordinates
    x_vals = [t[0] for t in all_tuples]
    y_vals = [t[1] for t in all_tuples]

    sns.scatterplot(x=x_vals, y=y_vals, ax=axes[0])
    axes[0].set_xlabel('X coordinate')
    axes[0].set_ylabel('Y coordinate')
    axes[0].set_title("Centroids")
    axes[0].invert_yaxis()  # Invert y-axis to match image coordinate system

    # Plot top left x1,y1 coordinates
    all_x1 = [x[0] for x in df['x_1'] for _ in range(len(df['x_1']))]
    all_y1 = [y[0] for y in df['y_1'] for _ in range(len(df['y_1']))]
    sns.scatterplot(x=all_x1, y=all_y1, ax=axes[1], color='orange')
    axes[1].set_xlabel('X1 (top-left)')
    axes[1].set_ylabel('Y1 (top-left)')
    axes[1].set_title("Top-Left Corners")
    axes[1].invert_yaxis()  # Invert y-axis to match image coordinate system

    # Plot top left x2,y2 coordinates
    all_x2 = [x[1] for x in df['x_2'] for _ in range(len(df['x_2']))]
    all_y2 = [y[1] for y in df['y_2'] for _ in range(len(df['y_2']))]
    sns.scatterplot(x=all_x2, y=all_y2, ax=axes[2], color='green')
    axes[2].set_xlabel('X2 (bottom-right)')
    axes[2].set_ylabel('Y2 (bottom-right)')
    axes[2].set_title("Bottom-Right Corners")
    axes[2].invert_yaxis()  # Invert y-axis to match image coordinate system

    plt.show()

def get_average_hsv(image_path):
    """Get average H, S, V values for a single image"""
    img = cv.imread(image_path)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    # Calculate mean values for each channel
    avg_h = np.mean(hsv[:, :, 0])  # Hue
    avg_s = np.mean(hsv[:, :, 1])  # Saturation  
    avg_v = np.mean(hsv[:, :, 2])  # Value
    
    return avg_h, avg_s, avg_v

# Add colorful plotting
def plot_hsv_analysis(df):
    """Plot HSV analysis with colors including Value histogram"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Changed to 2x3 to fit Value histogram
    
    # 1. Bar chart of average values with appropriate colors
    channels = ['Hue', 'Saturation', 'Value']
    means = [df['avg_hue'].mean(), df['avg_saturation'].mean(), df['avg_value'].mean()]
    colors = ['orange', 'purple', 'gray']
    
    bars = axes[0,0].bar(channels, means, color=colors, alpha=0.7, edgecolor='black')
    axes[0,0].set_title('Dataset Average HSV Values')
    axes[0,0].set_ylabel('Average Value')
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        axes[0,0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                      f'{mean:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Hue distribution with rainbow colors
    n_bins = 30
    counts, bins, patches = axes[0,1].hist(df['avg_hue'], bins=n_bins, alpha=0.7, edgecolor='black')
    
    # Color each bar according to its hue value
    for i, (count, bin_edge, patch) in enumerate(zip(counts, bins, patches)):
        # Convert bin center to color (hue value 0-179 -> 0-1 for colormap)
        hue_normalized = (bins[i] + bins[i+1]) / 2 / 179
        patch.set_facecolor(plt.cm.hsv(hue_normalized))
    
    axes[0,1].set_title('Average Hue Distribution')
    axes[0,1].set_xlabel('Average Hue (0-179)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].axvline(df['avg_hue'].mean(), color='red', linestyle='--', 
                     linewidth=2, label=f'Mean: {df["avg_hue"].mean():.1f}')
    axes[0,1].legend()
    
    # 3. Saturation distribution (grayscale to vibrant)
    counts, bins, patches = axes[0,2].hist(df['avg_saturation'], bins=n_bins, alpha=0.7, edgecolor='black')
    
    # Color bars from gray (low sat) to vibrant (high sat)
    for i, (count, bin_edge, patch) in enumerate(zip(counts, bins, patches)):
        sat_normalized = (bins[i] + bins[i+1]) / 2 / 255
        # Use a color that goes from gray to vibrant
        color = plt.cm.viridis(sat_normalized)
        patch.set_facecolor(color)
    
    axes[0,2].set_title('Average Saturation Distribution')
    axes[0,2].set_xlabel('Average Saturation (0-255)')
    axes[0,2].set_ylabel('Frequency')
    axes[0,2].axvline(df['avg_saturation'].mean(), color='red', linestyle='--',
                     linewidth=2, label=f'Mean: {df["avg_saturation"].mean():.1f}')
    axes[0,2].legend()
    
    # 4. NEW: Value distribution (black to white gradient)
    counts, bins, patches = axes[1,0].hist(df['avg_value'], bins=n_bins, alpha=0.7, edgecolor='black')
    
    # Color bars from black (low value) to white (high value)
    for i, (count, bin_edge, patch) in enumerate(zip(counts, bins, patches)):
        value_normalized = (bins[i] + bins[i+1]) / 2 / 255
        # Use grayscale colormap (black to white)
        color = plt.cm.gray(value_normalized)
        patch.set_facecolor(color)
    
    axes[1,0].set_title('Average Value (Brightness) Distribution')
    axes[1,0].set_xlabel('Average Value (0-255)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].axvline(df['avg_value'].mean(), color='red', linestyle='--',
                     linewidth=2, label=f'Mean: {df["avg_value"].mean():.1f}')
    axes[1,0].legend()
    
    # 5. 2D scatter: Hue vs Saturation colored by Value
    scatter = axes[1,1].scatter(df['avg_hue'], df['avg_saturation'], 
                               c=df['avg_value'], cmap='plasma', 
                               alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    axes[1,1].set_xlabel('Average Hue')
    axes[1,1].set_ylabel('Average Saturation')
    axes[1,1].set_title('Hue vs Saturation (colored by Value/Brightness)')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[1,1])
    cbar.set_label('Average Value (Brightness)')
    
    # Add dataset mean point
    axes[1,1].scatter(df['avg_hue'].mean(), df['avg_saturation'].mean(), 
                     color='red', s=100, marker='x', linewidth=3, 
                     label='Dataset Mean')
    axes[1,1].legend()
    
    # 6. NEW: Additional scatter plot - Saturation vs Value colored by Hue
    scatter2 = axes[1,2].scatter(df['avg_saturation'], df['avg_value'], 
                                c=df['avg_hue'], cmap='hsv', 
                                alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    axes[1,2].set_xlabel('Average Saturation')
    axes[1,2].set_ylabel('Average Value (Brightness)')
    axes[1,2].set_title('Saturation vs Value (colored by Hue)')
    
    # Add colorbar for hue
    cbar2 = plt.colorbar(scatter2, ax=axes[1,2])
    cbar2.set_label('Average Hue')
    
    # Add dataset mean point
    axes[1,2].scatter(df['avg_saturation'].mean(), df['avg_value'].mean(), 
                     color='red', s=100, marker='x', linewidth=3, 
                     label='Dataset Mean')
    axes[1,2].legend()
    
    plt.tight_layout()
    plt.show()

input_dir = r'C:\Users\anamk\projects\wildlife_tracker\image\yolo_bird_data'

# Create dataframe with detection info
df = data_clean.create_df(input_dir, delim='_')
plot_coords(df)
breakpoint()
plot_set_2(df)

# Filter for single detections AND any area < 100
filtered_df = df[(df['detections'] == 1) & 
                 (df['bbox_area'].apply(lambda x: any(area < 300 and area > 250 for area in x)))]
breakpoint()
# Average HSV histogram for all images
hsv_data = df['jpg_files'].apply(get_average_hsv)
df[['avg_hue', 'avg_saturation', 'avg_value']] = pd.DataFrame(hsv_data.tolist(), index=df.index)

# Calculate dataset-wide averages
dataset_avg_h = df['avg_hue'].mean()
dataset_avg_s = df['avg_saturation'].mean()
dataset_avg_v = df['avg_value'].mean()

print(f"Dataset Average HSV: H={dataset_avg_h:.1f}, S={dataset_avg_s:.1f}, V={dataset_avg_v:.1f}")
# Hue (H) → the "type of color", “Is it red, blue, green, etc.?”
# Saturation (S) → the "intensity of color", “Is it a vivid color or more washed out?”
    # High saturation → pure, vibrant color (like neon green).
    # Low saturation → washed-out, grayish color.
# Value (V) → the "brightness of color", “Is it a dark color or a light color?”
    # High value → bright (close to white).
    # Low value → dark (close to black).

plot_hsv_analysis(df)
