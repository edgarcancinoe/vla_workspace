
import numpy as np
import cv2

# =============================================================================
# CAMERA CALIBRATION MATRICES
# =============================================================================
# Replace these with your actual calibration values!

# Camera 1 (Top? Or Lateral? Check your config)
CAM1_MATRIX = np.array([
    [373.07230519, 0.00000000, 316.82237586],
    [0.00000000, 373.06048512, 244.77337449],
    [0.00000000, 0.00000000, 1.00000000]
])
CAM1_DIST_COEFFS = np.array([-0.21391238, 0.07484612, 0.00098082, -0.00179784, -0.01754548])

# Camera 2
CAM0_MATRIX = np.array([
    [377.27012802, 0.00000000, 366.14798850],
    [0.00000000, 376.48989004, 243.83856485],
    [0.00000000, 0.00000000, 1.00000000]
])
CAM0_DIST_COEFFS = np.array([-0.21314048, 0.05494787, -0.00028633, -0.00023334, 0.00063803])

# Map camera names (as used in robot config) to matrices
# In launch_client.yaml you have 'camera1' and 'camera2' keys (before rename_map)
# But wait, rename_map patches the keys in the OBSERVATION dict.
# The `raw_observation` keys come from the robot config keys.
# In your current launch_client.yaml, keys are "camera1" and "camera2".
# So `get_observation` will returning {"camera1": img, "camera2": img}.

CAMERA_CALIBRATION = {
    "camera1": {"mtx": CAM1_MATRIX, "dist": CAM1_DIST_COEFFS},
    "camera2": {"mtx": CAM0_MATRIX, "dist": CAM0_DIST_COEFFS},
    # Also aliases if needed
    "top": {"mtx": CAM1_MATRIX, "dist": CAM1_DIST_COEFFS}, 
    "lateral": {"mtx": CAM0_MATRIX, "dist": CAM0_DIST_COEFFS},
    "wrist": {"mtx": CAM0_MATRIX, "dist": CAM0_DIST_COEFFS},
}

def rectify_image(image, camera_name):
    """Undistorts an image using the calibration for the given camera name."""
    if camera_name not in CAMERA_CALIBRATION:
        return image
    
    calib = CAMERA_CALIBRATION[camera_name]
    mtx = calib["mtx"]
    dist = calib["dist"]
    
    h, w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    # undistort
    dst = cv2.undistort(image, mtx, dist, None, newcameramtx)
    
    # crop the image
    x, y, w, h = roi
    if w > 0 and h > 0: # Ensure valid roi
        dst = dst[y:y+h, x:x+w]
        
    # Resize back to original size (640x480) as models/datasets expect fixed size
    if dst.shape[0] != 480 or dst.shape[1] != 640:
        dst = cv2.resize(dst, (640, 480))
        
    return dst
