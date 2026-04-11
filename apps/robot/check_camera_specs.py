import cv2

def check_camera_specs(index_range=range(4)):
    print("Scanning for cameras...")
    for index in index_range:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            continue

        print(f"\n--- Camera Index: {index} ---")
        backend = cap.getBackendName()
        print(f"Backend: {backend}")

        # Get current/default settings
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Default Configuration: {int(width)}x{int(height)} @ {fps} FPS")

        # Test common resolutions
        print("Testing Resolutions:")
        test_resolutions = [
            (3840, 2160), # 4K
            (1920, 1080), # 1080p
            (1280, 720),  # 720p
            (640, 480)    # 480p
        ]

        for w, h in test_resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            
            # Read back actual values
            actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            
            supported = (actual_w == w and actual_h == h)
            status = "Supported" if supported else f"No (Got {int(actual_w)}x{int(actual_h)})"
            
            if supported:
                print(f"  {w}x{h}: {status} @ {actual_fps} FPS")
            
        cap.release()
    print("\nScan complete.")

if __name__ == "__main__":
    check_camera_specs()
