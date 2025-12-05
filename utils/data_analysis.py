import data_gen as data_gen
import seaborn as sns
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
import numpy as np

def plot_set_1(df):
    # Plot areas captured and detection times
    #TODO: This is only the MAX area detected in each frame, not all bounding boxes
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    sns.histplot(data=df, x='max_areas', kde=True, ax=axes[0,0])
    axes[0,0].set_title("Areas")

    sns.scatterplot(data=df, x='hours', y='max_areas', ax = axes[0,1])
    axes[0,1].set_title('Hours vs. Areas')

    sns.scatterplot(data=df, x='hours', y='detections', ax = axes[1,0])
    axes[1,0].set_title('Hour vs. Detections')

    sns.histplot(data=df, x='detections', kde=True, ax = axes[1,1])
    axes[1,1].set_title('Detections')

    plt.show()

def plot_set_2(df):
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
    fig, axes = plt.subplots(1, 2, figsize=(10, 8))

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
    all_x1 = [x for x_list in df['bbox_x1'] for x in x_list]
    all_y1 = [y for y_list in df['bbox_y1'] for y in y_list]
    top_left = pd.DataFrame({'x': all_x1, 'y': all_y1, 'label': 'top-left'})

    # Plot bottom right x2,y2 coordinates
    all_x2 = [x for x_list in df['bbox_x2'] for x in x_list]
    all_y2 = [y for y_list in df['bbox_y2'] for y in y_list]
    bottom_right = pd.DataFrame({'x': all_x2, 'y': all_y2, 'label': 'bottom-right'})
    combined = pd.concat([top_left, bottom_right], ignore_index=True)
    
    sns.scatterplot(data=combined, x='x', y='y', hue='label', ax=axes[1])
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_title("Bounding Box Corners")
    plt.ylim(0, 1080)
    plt.xlim(0, 1920)
    axes[1].invert_yaxis()  # Invert y-axis to match image coordinate system


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

def df_lum(files):

    dict_list = []
    for file in files:
        img = cv.imread(str(file))
        if img is None:
            return None
            
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        
        # Basic statistics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        min_val = np.min(gray)
        max_val = np.max(gray)
        
        # Percentiles for exposure analysis
        p1 = np.percentile(gray, 1)
        p5 = np.percentile(gray, 5)
        p25 = np.percentile(gray, 25)
        p50 = np.percentile(gray, 50)  # median
        p75 = np.percentile(gray, 75)
        p95 = np.percentile(gray, 95)
        p99 = np.percentile(gray, 99)
        
        # Contrast metrics
        # RMS contrast (standard deviation)
        rms_contrast = std_brightness
        
        # Michelson contrast (for images with distinct light/dark regions)
        michelson_contrast = (max_val - min_val) / (max_val + min_val) if (max_val + min_val) > 0 else 0
        
        # Weber contrast (relative to background)
        weber_contrast = std_brightness / mean_brightness if mean_brightness > 0 else 0
        
        # Histogram analysis
        hist = cv.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        
        # Exposure indicators
        underexposed_pixels = np.sum(gray < 30) / gray.size  # Percentage of very dark pixels
        overexposed_pixels = np.sum(gray > 225) / gray.size  # Percentage of very bright pixels
        clipped_shadows = np.sum(gray == 0) / gray.size  # Pure black pixels
        clipped_highlights = np.sum(gray == 255) / gray.size  # Pure white pixels
        
        # Dynamic range
        dynamic_range = p99 - p1
        
        # Histogram spread metrics
        histogram_entropy = -np.sum((hist/np.sum(hist)) * np.log2((hist/np.sum(hist)) + 1e-10))
        
        # Low-key vs high-key detection
        low_key_ratio = np.sum(gray < 85) / gray.size  # Dark image indicator
        high_key_ratio = np.sum(gray > 170) / gray.size  # Bright image indicator

        # Check for underexposure
        issues = []
        thresholds = {
            'very_dark': 50,        # Mean brightness threshold for very dark images
            'very_bright': 200,     # Mean brightness threshold for very bright images
            'low_contrast': 30,     # RMS contrast threshold for low contrast
            'high_overexposure': 0.05,  # 5% overexposed pixels
            'high_underexposure': 0.05,  # 5% underexposed pixels
            'clipping_threshold': 0.01    # 1% clipped pixels
        }

        if mean_brightness < thresholds['very_dark']:
            issues.append('very_dark')
        if underexposed_pixels > thresholds['high_underexposure']:
            issues.append('underexposed')
        if clipped_shadows > thresholds['clipping_threshold']:
            issues.append('shadow_clipping')

        # Check for overexposure
        if mean_brightness > thresholds['very_bright']:
            issues.append('very_bright')
        if overexposed_pixels > thresholds['high_overexposure']:
            issues.append('overexposed')
        if clipped_highlights > thresholds['clipping_threshold']:
            issues.append('highlight_clipping')
        
        # Check for low contrast
        if rms_contrast < thresholds['low_contrast']:
            issues.append('low_contrast')
        
        # Check for extreme lighting bias
        if low_key_ratio > 0.8:
            issues.append('extreme_low_key')
        if high_key_ratio > 0.8:
            issues.append('extreme_high_key')

        bright_contr_dict = {
                'filename': file,
                'issues': issues,
                'mean_brightness': mean_brightness,
                'std_brightness': std_brightness,
                'min_luminance': min_val,
                'max_luminance': max_val,
                'median_brightness': p50,
                'p1': p1,
                'p5': p5,
                'p25': p25,
                'p75': p75,
                'p95': p95,
                'p99': p99,
                'rms_contrast': rms_contrast,
                'michelson_contrast': michelson_contrast,
                'weber_contrast': weber_contrast,
                'dynamic_range': dynamic_range,
                'histogram_entropy': histogram_entropy,
                'underexposed_ratio': underexposed_pixels,
                'overexposed_ratio': overexposed_pixels,
                'clipped_shadows': clipped_shadows,
                'clipped_highlights': clipped_highlights,
                'low_key_ratio': low_key_ratio,
                'high_key_ratio': high_key_ratio
            }

        dict_list.append(bright_contr_dict)

    return pd.DataFrame(dict_list)

def plot_distribution_analysis(df):
    """Create visualization plots for the analysis with KDE overlays."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Brightness distribution
    axes[0, 0].hist(df['mean_brightness'], bins=50, alpha=0.5, color='skyblue', density=True, label='Histogram')
    sns.kdeplot(data=df, x='mean_brightness', ax=axes[0, 0], color='darkblue', linewidth=2, label='KDE')
    axes[0, 0].axvline(df['mean_brightness'].mean(), color='red', linestyle='--', label=f'Mean ({df["mean_brightness"].mean():.1f})')
    axes[0, 0].axvline(df['mean_brightness'].median(), color='orange', linestyle=':', label=f'Median ({df["mean_brightness"].median():.1f})')
    axes[0, 0].set_title('Brightness Distribution')
    axes[0, 0].set_xlabel('Mean Brightness')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    
    # Contrast distribution
    axes[0, 1].hist(df['rms_contrast'], bins=50, alpha=0.5, color='lightgreen', density=True, label='Histogram')
    sns.kdeplot(data=df, x='rms_contrast', ax=axes[0, 1], color='darkgreen', linewidth=2, label='KDE')
    axes[0, 1].axvline(df['rms_contrast'].mean(), color='red', linestyle='--', label=f'Mean ({df["rms_contrast"].mean():.1f})')
    axes[0, 1].axvline(df['rms_contrast'].median(), color='orange', linestyle=':', label=f'Median ({df["rms_contrast"].median():.1f})')
    axes[0, 1].set_title('Contrast Distribution (RMS)')
    axes[0, 1].set_xlabel('RMS Contrast')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    
    # Dynamic range
    axes[0, 2].hist(df['dynamic_range'], bins=50, alpha=0.5, color='orange', density=True, label='Histogram')
    sns.kdeplot(data=df, x='dynamic_range', ax=axes[0, 2], color='darkorange', linewidth=2, label='KDE')
    axes[0, 2].axvline(df['dynamic_range'].mean(), color='red', linestyle='--', label=f'Mean ({df["dynamic_range"].mean():.1f})')
    axes[0, 2].axvline(df['dynamic_range'].median(), color='blue', linestyle=':', label=f'Median ({df["dynamic_range"].median():.1f})')
    axes[0, 2].set_title('Dynamic Range Distribution')
    axes[0, 2].set_xlabel('Dynamic Range (P99-P1)')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].legend()
    
    # Exposure problems
    axes[1, 0].scatter(df['overexposed_ratio'], df['underexposed_ratio'], alpha=0.6, color='coral')
    axes[1, 0].set_xlabel('Overexposed Ratio')
    axes[1, 0].set_ylabel('Underexposed Ratio')
    axes[1, 0].set_title('Exposure Problems')
    # Add quadrant lines to help interpretation
    axes[1, 0].axhline(0.05, color='red', linestyle=':', alpha=0.5, label='5% threshold')
    axes[1, 0].axvline(0.05, color='red', linestyle=':', alpha=0.5)
    axes[1, 0].legend()
    
    # Brightness vs Contrast with density
    scatter = axes[1, 1].scatter(df['mean_brightness'], df['rms_contrast'], alpha=0.6, c=df['dynamic_range'], 
                                cmap='viridis', s=30)
    axes[1, 1].set_xlabel('Mean Brightness')
    axes[1, 1].set_ylabel('RMS Contrast')
    axes[1, 1].set_title('Brightness vs Contrast (colored by Dynamic Range)')
    plt.colorbar(scatter, ax=axes[1, 1], label='Dynamic Range')
    
    # Histogram entropy
    axes[1, 2].hist(df['histogram_entropy'], bins=50, alpha=0.5, color='purple', density=True, label='Histogram')
    sns.kdeplot(data=df, x='histogram_entropy', ax=axes[1, 2], color='darkviolet', linewidth=2, label='KDE')
    axes[1, 2].axvline(df['histogram_entropy'].mean(), color='red', linestyle='--', label=f'Mean ({df["histogram_entropy"].mean():.1f})')
    axes[1, 2].axvline(df['histogram_entropy'].median(), color='orange', linestyle=':', label=f'Median ({df["histogram_entropy"].median():.1f})')
    axes[1, 2].set_title('Histogram Entropy Distribution')
    axes[1, 2].set_xlabel('Entropy')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.show()


def analysis():
    input_dir = r"C:\Users\anamk\projects\dataSets\birds.v4i.yolov11"
    # Create dataframe with detection info
    df = data_gen.create_df(input_dir, delim='_')

    # Plot info on detection area, dections, time of day
    plot_set_1(df)

    # Plot info on AR, image size and classes
    plot_set_2(df)

    birds_only = df[df['class'] == 'bird']
    plot_coords(birds_only)

    # Brightness and contrast analysis
    df_lum_set = df_lum(df['jpg_files'])
    plot_distribution_analysis(df_lum_set)

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

    # Filter for single detections AND any area < 100
    filtered_df = df[(df['detections'] == 1) & 
                    (df['bbox_area'].apply(lambda x: any(area < 300 and area > 250 for area in x)))]


if __name__ == "__main__":
    analysis()