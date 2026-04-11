
import cv2
import numpy as np
import argparse
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Calibrate camera using a checkerboard.")
    parser.add_argument("--camera-index", type=int, default=1, help="Camera index to calibrate (0, 1, ...)")
    parser.add_argument("--rows", type=int, default=8, help="Number of inner corners per row (usually squares - 1)")
    parser.add_argument("--cols", type=int, default=6, help="Number of inner corners per column (usually squares - 1)")
    parser.add_argument("--square-size", type=float, default=0.030, help="Size of one square in meters (e.g. 0.030 for 30mm)")
    parser.add_argument("--output", type=str, default="calibration_result.txt", help="File to save results")
    args = parser.parse_args()

    # Checkerboard dimensions (inner corners)
    CHECKERBOARD = (args.rows, args.cols)
    SQUARE_SIZE = args.square_size

    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
    # Z is 0 for the board plane
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE

    # Arrays to store object points and image points from all valid images
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane

    print(f"Connecting to camera {args.camera_index}...")
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Set resolution if possible (match your robot config, usually 640x480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nControls:")
    print("  'c' or SPACE: Capture frame (if corners detected)")
    print("  'q': Finish and Calibrate")
    print("  'r': Reset/Clear captured points\n")

    full_coverage = False
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find chess board corners
        ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        display_frame = frame.copy()

        # If found, draw corners
        if ret_corners:
            # Refine corners
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners2, ret_corners)
            
            # Text status
            cv2.putText(display_frame, "Pattern Detected! Press SPACE to capture", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
             cv2.putText(display_frame, "Searching for pattern...", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(display_frame, f"Captured: {count} frames", (20, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow('Camera Calibration', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c') or key == 32: # 'c' or SPACE
            if ret_corners:
                objpoints.append(objp)
                imgpoints.append(corners2)
                count += 1
                print(f"Captured frame {count}")
                # Flash effect
                cv2.imshow('Camera Calibration', 255 - display_frame)
                cv2.waitKey(50)
            else:
                print("Pattern not detected, cannot capture.")
        elif key == ord('r'):
            objpoints = []
            imgpoints = []
            count = 0
            print("Reset captured points.")

    cap.release()
    cv2.destroyAllWindows()

    if count < 10:
        print("Not enough frames captured for good calibration (need > 10, preferably > 20).")
        return

    print("\nCalculating calibration... (this may take a moment)")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("\n" + "="*60)
    print(f"CALIBRATION RESULT (Reprojection Error: {ret:.4f})")
    print("="*60)
    
    # Format code block for copy-paste
    code_block = f"""
# Camera Index {args.camera_index}
CAM{args.camera_index}_MATRIX = np.array([
    [{mtx[0,0]:.8f}, {mtx[0,1]:.8f}, {mtx[0,2]:.8f}],
    [{mtx[1,0]:.8f}, {mtx[1,1]:.8f}, {mtx[1,2]:.8f}],
    [{mtx[2,0]:.8f}, {mtx[2,1]:.8f}, {mtx[2,2]:.8f}]
])
CAM{args.camera_index}_DIST_COEFFS = np.array([{', '.join([f'{x:.8f}' for x in dist[0]])}])
"""
    print(code_block)
    print("="*60)
    
    # Save to file
    with open(args.output, "a") as f:
        f.write(code_block + "\n")
    print(f"Results appended to {args.output}")

if __name__ == "__main__":
    main()
