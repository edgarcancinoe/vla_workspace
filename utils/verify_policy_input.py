
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
import cv2

# Adjust path to find lerobot
sys.path.append("/Users/edgarcancino/Documents/Academic/EMAI Thesis/repos/lerobot/src")

from lerobot.async_inference.robot_client import RobotClient, RobotClientConfig
import draccus.parsers.decoding as draccus_decoding

# Load config
LAUNCH_CONFIG_PATH = Path("config/launch_client.yaml")
ROBOT_CONFIG_PATH = Path("config/robot_config.yaml")

def main():
    print("--- Verifying Policy Input Color Space ---")
    
    # 1. Load Config
    with open(LAUNCH_CONFIG_PATH, 'r') as f:
        launch_cfg_dict = yaml.safe_load(f)
    if ROBOT_CONFIG_PATH.exists():
        with open(ROBOT_CONFIG_PATH, 'r') as f:
            robot_cfg_dict = yaml.safe_load(f)
        if "robot" in robot_cfg_dict and "port" in robot_cfg_dict["robot"]:
            launch_cfg_dict["robot"]["port"] = robot_cfg_dict["robot"]["port"]
            
    cfg = draccus_decoding.decode(RobotClientConfig, launch_cfg_dict)
    
    # 2. Initialize RobotClient (connects to cameras)
    print("Initializing Robot (checking cameras)...")
    # Disable Rerun for this check
    cfg.use_rerun = False
    client = RobotClient(cfg)
    
    # 3. Capture one observation
    obs = client.robot.get_observation()
    
    # 4. Check Image
    print("\n[Analysis]")
    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            print(f"Checking {key}: Shape {value.shape}")
            
            # Assuming value is [H, W, C] (OpenCV standard) or [C, H, W] (Tensor output)
            # Robot.get_observation usually returns what the camera class returns.
            # OpenCVCamera returns [C, H, W] or [H, W, C]? Let's check shape.
            
            if value.shape[0] == 3:
                # [C, H, W]
                img_hwc = np.transpose(value, (1, 2, 0))
                print("  Detected [C, H, W], transposed to [H, W, C]")
            else:
                img_hwc = value
                
            # Take center pixel
            h, w, c = img_hwc.shape
            center_pixel = img_hwc[h//2, w//2]
            print(f"  Center Pixel Value: {center_pixel}")
            
            # Save for inspection
            # If data is RGB, cv2.imwrite (expecting BGR) will swap it.
            # So if we save as 'verify_policy_input.png' and it looks BLUE, then data was RGB.
            # If it looks ORANGE, then data was BGR.
            
            # Save raw
            cv2.imwrite(f"verify_policy_{key}.png", img_hwc)
            print(f"  Saved 'verify_policy_{key}.png'")
            print("  -> If this image looks BLUE (swapped colors), then the input is RGB (Correct).")
            print("  -> If this image looks NORMAL, then the input is BGR (Incorrect for policy).")

    client.stop()

if __name__ == "__main__":
    main()
