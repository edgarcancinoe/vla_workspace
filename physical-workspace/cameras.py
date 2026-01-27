import cv2
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

# Define the cameras you want to use (e.g., indices 0 and 1)
CAMERA_INDICES = [1, 2] 
res = (640, 480)
cameras = []

try:
    # Initialize and connect to all cameras
    for idx in CAMERA_INDICES:
        print(f"Connecting to camera {idx}...")
        config = OpenCVCameraConfig(
            index_or_path=idx,
            fps=30,
            width=res[0],  
            height=res[1],
            color_mode=ColorMode.RGB,
            rotation=Cv2Rotation.NO_ROTATION
        )
        cam = OpenCVCamera(config)
        cam.connect()
        cameras.append(cam)
        print(f"Camera {idx} connected.")

    print("Press 'q' to quit.")

    count = 0
    while count < 100:
        for i, cam in enumerate(cameras):
            # Read frame (returns RGB)
            frame = cam.async_read()
            
            if frame is not None:
                # Convert RGB (LeRobot default) to BGR for OpenCV display
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Show the frame
                window_name = f"Camera {CAMERA_INDICES[i]}"
                # cv2.imshow(window_name, frame_bgr) # Commented out for headless run safety if needed
        
        count += 1
        # Check for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("Disconnecting cameras...")
    for cam in cameras:
        cam.disconnect()
    cv2.destroyAllWindows()
    print("Done.")