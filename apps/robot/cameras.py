import cv2
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

CAMERA_INDICES = [0,1]   # Add more indices if needed, e.g. [0, 1]
res = (640, 480)
cameras = []

try:
    # Initialize cameras
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

    while True:
        for i, cam in enumerate(cameras):
            frame = cam.async_read()

            if frame is not None:
                # Convert RGB → BGR for OpenCV display
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Overlay camera index text
                cv2.putText(
                    frame_bgr,
                    f"Camera {CAMERA_INDICES[i]}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                cv2.imshow(f"Camera {CAMERA_INDICES[i]}", frame_bgr)

        # Needed for window refresh
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("Disconnecting cameras...")
    for cam in cameras:
        cam.disconnect()
    cv2.destroyAllWindows()
    print("Done.")