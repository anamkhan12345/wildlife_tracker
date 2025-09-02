import cv2

def check_webcam_resolution(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    # Get current resolution
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Current Resolution: {int(width)}x{int(height)}")
    print(f"FPS: {fps}")
    
    # Test different resolutions to see what's supported
    test_resolutions = [
        (1280, 720),  # HD
        (1280, 1024), # SXGA
        (1920, 1080) # Full HD
    ]
    
    print("\nSupported resolutions:")
    for w, h in test_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_w == w and actual_h == h:
            print(f"  ✓ {w}x{h}")
        else:
            print(f"  ✗ {w}x{h} -> {actual_w}x{actual_h}")
    
    cap.release()

# Check your webcam
check_webcam_resolution(0)  # Change to 1, 2, etc. for other cameras
