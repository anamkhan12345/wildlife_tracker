import os
import re
import glob
from datetime import datetime

def convert_datetime_to_timestamp(date_str, time_str):
    """
    Convert YYYYMMDD and HHMMSS to Unix timestamp in milliseconds
    """
    try:
        # Parse date and time
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])
        
        # Create datetime object
        dt = datetime(year, month, day, hour, minute, second)
        
        # Convert to Unix timestamp in milliseconds
        timestamp_ms = int(dt.timestamp() * 1000)
        return str(timestamp_ms)
    except Exception as e:
        print(f"Error converting datetime: {e}")
        return None

def transform_filename(filename):
    """
    Transform filename according to the patterns:
    A→B: timestamp_area_x_y → detect_timestamp_area_x_y
    C→D: negative_YYYYMMDD_HHMMSS_n → negative_timestamp_area_0_n
    """
    
    # Remove file extension for processing
    dir_name = os.path.dirname(filename)
    name_without_ext = os.path.splitext(os.path.basename(filename))[0]
    extension = os.path.splitext(filename)[1]

    # Pattern A: timestamp_area_x_y → detect_timestamp_area_x_y
    pattern_a = r'^(\d+)_area_(\d+)_(\d+)$'
    match_a = re.match(pattern_a, name_without_ext)
    
    if match_a:
        timestamp, x_coord, y_coord = match_a.groups()
        new_name = f"detect_{timestamp}_area_{x_coord}_{y_coord}"
        return os.path.join(dir_name, new_name + extension)
    
    # Pattern C: negative_YYYYMMDD_HHMMSS_n → negative_timestamp_area_0_n
    pattern_c = r'^negative_(\d{8})_(\d{6})_(\d+)$'
    match_c = re.match(pattern_c, name_without_ext)
    
    if match_c:
        date_part, time_part, number = match_c.groups()
        timestamp = convert_datetime_to_timestamp(date_part, time_part)
        if timestamp:
            new_name = f"negative_{timestamp}_area_0_{number}"
            return os.path.join(dir_name, new_name + extension)
        else:
            print(f"Failed to convert timestamp for {filename}")
            return filename
    
    # If no pattern matches, return original filename
    print(f"No matching pattern for: {filename}")
    return filename

def batch_rename_files(folder_path, dry_run=True):
    """
    Batch rename files in a folder
    
    Args:
        folder_path: Path to the folder containing files
        dry_run: If True, only show what would be renamed without actually renaming
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        return
    
    files = glob.glob(f"{folder_path}/**/*", recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    rename_operations = []
    
    for filename in files:
        new_filename = transform_filename(filename)
        
        if new_filename != filename:
            rename_operations.append((filename, new_filename))
    
    if not rename_operations:
        print("No files match the transformation patterns.")
        return
    
    print(f"Found {len(rename_operations)} files to rename:")
    print("-" * 80)
    
    for old_name, new_name in rename_operations:
        print(f"{old_name} → {new_name}")
    
    if dry_run:
        print("-" * 80)
        print("This was a dry run. To actually rename files, set dry_run=False")
        return
    
    # Actually rename files
    renamed_count = 0
    for old_name, new_name in rename_operations:
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)
        
        # Check if new filename already exists
        if os.path.exists(new_path):
            print(f"Warning: {new_name} already exists, skipping {old_name}")
            continue
            
        try:
            os.rename(old_path, new_path)
            print(f"✓ Renamed: {old_name}")
            renamed_count += 1
        except Exception as e:
            print(f"✗ Error renaming {old_name}: {e}")
    
    print(f"\nTotal files renamed: {renamed_count}")

# Example usage and testing
if __name__ == "__main__":
    # # Test the transformations
    # test_files = [
    #     "1755784844027_area_54_878.jpg",
    #     "detect_1755784844027_area_54_878.txt",
    #     "negative_20250829_180139_9.txt",
    #     "negative_20250829_180139_9.jpg",
    #     "negative_1756490198958_area_0_8.jpg",
    #     "detect_1755784844027_area_54_878.jpg",
    #     "some_other_file.txt"  # This should not match any pattern
    # ]
    
    # print("Testing transformations:")
    # print("=" * 80)
    # for test_file in test_files:
    #     result = transform_filename(test_file)
    #     status = "✓" if result != test_file else "○"
    #     print(f"{status} {test_file} → {result}")
    
    # print("\n" + "=" * 80)
    # print("To rename files in your folder:")
    # print("1. Set folder_path to your directory")
    # print("2. Run with dry_run=True first to preview")
    # print("3. Run with dry_run=False to actually rename")
    # print("Example:")
    # print('folder_path = "/path/to/your/folder"')
    # print('batch_rename_files(folder_path, dry_run=True)   # Preview')
    # print('batch_rename_files(folder_path, dry_run=False)  # Execute')

    batch_rename_files(r"C:\Users\anamk\projects\wildlife_tracker\image\yolo_set\test", dry_run=False)
    batch_rename_files(r"C:\Users\anamk\projects\wildlife_tracker\image\yolo_set\train", dry_run=False)
    batch_rename_files(r"C:\Users\anamk\projects\wildlife_tracker\image\yolo_set\val", dry_run=False)
