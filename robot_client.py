import time
import requests
import torch
import numpy as np
import rerun as rr
from dataclasses import dataclass
from typing import Dict, Optional, Any

from pathlib import Path
# LeRobot Imports
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.cameras.opencv import OpenCVCameraConfig

# --- CONFIGURATION ---
@dataclass
class ClientConfig:
    # Network
    server_url: str = "http://localhost:8080/forward"  # Tunneled to Cluster
    timeout: float = 2.0                              # Max wait for server response in seconds
    
    # Robot Hardware (SO-101 / Feetech)
    robot_type: str = "so101_follower"
    robot_port: str = "/dev/cu.usbmodem5AB90655421"
    camera_id: int = 0
    
    # Control Loop
    control_frequency: int = 30  # Hz
    max_safety_lag: float = 0.5  # Stop if last packet was > 0.5s ago
    
    # Visualization
    use_rerun: bool = True

class RemotePolicy:
    """
    Acts as the 'Brain' proxy. 
    It formats observations, sends them to the cluster, and decodes the action.
    """
    def __init__(self, url: str, timeout: float):
        self.url = url
        self.timeout = timeout
        self.session = requests.Session() # Reuse connection for speed

    def select_action(self, observation: Dict[str, Any]) -> torch.Tensor:
        """
        Sends observation dict to server, returns action tensor.
        Uses JPEG compression for images to avoid massive JSON payloads.
        """
        import base64
        import cv2
        import numpy as np

        payload = {}
        for k, v in observation.items():
            # If it's the image, compress it to JPEG so the network doesn't choke
            if k == "laptop" and isinstance(v, np.ndarray):
                # Convert BGR (OpenCV) to JPEG
                _, buffer = cv2.imencode('.jpg', v, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                payload[k] = base64.b64encode(buffer).decode('utf-8')
                payload[f"{k}_format"] = "jpeg_base64"
            
            elif isinstance(v, torch.Tensor):
                payload[k] = v.tolist()
            elif isinstance(v, np.ndarray):
                payload[k] = v.tolist()
            else:
                payload[k] = v

        try:
            start_time = time.time()
            # 2. Send to Cluster
            response = self.session.post(self.url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            latency = (time.time() - start_time) * 1000
            print(f"Brain Latency: {latency:.1f}ms")

            # 3. Decode Action
            data = response.json()
            return torch.tensor(data["action"])
            
        except requests.exceptions.RequestException as e:
            print(f"Network/Brain Error: {e}")
            return None

def main():
    cfg = ClientConfig()
    
    # 1. Initialize Visualization
    if cfg.use_rerun:
        rr.init("RobotClient", spawn=True)

    # 2. Initialize Hardware
    print(f"Connecting to Robot at {cfg.robot_port}...")
    
    # Create configuration for SO-101 Follower
    robot_config = SO101FollowerConfig(
        port=cfg.robot_port,
        calibration_dir=Path(".cache/calibration"),
        cameras={
            "laptop": OpenCVCameraConfig(
                index_or_path=cfg.camera_id, 
                fps=30, 
                width=640, 
                height=480
            )
        }
    )
    
    robot = SO101Follower(robot_config)
    robot.connect()
    print("Robot Connected.")

    # 3. Initialize Remote Brain
    policy = RemotePolicy(cfg.server_url, cfg.timeout)
    print(f"Linking to Brain at {cfg.server_url}...")

    # 4. State Variables
    action_queue = []
    last_packet_time = time.time()
    dt = 1.0 / cfg.control_frequency

    try:
        print("Starting Control Loop.")
        while True:
            loop_start = time.time()

            # --- A. CAPTURE ---
            # Read images and joint positions
            observation = robot.get_observation()
            capture_timestamp = time.time()
            
            # --- B. NETWORK REQUEST (ASYNC STRATEGY) ---
            # In a true async setup, this runs in a separate thread.
            # Here, we do a blocking call for simplicity, relying on the
            # Server's "Chunking" to give us a buffer.
            
            # Only request new plans if our buffer is running low
            if len(action_queue) < 5: 
                new_actions = policy.select_action(observation)
                
                if new_actions is not None:
                    last_packet_time = time.time()
                    # Append new actions to our local buffer
                    # Assuming new_actions is shape [Chunk_Size, Dims]
                    action_queue.extend(new_actions) 
            
            # --- C. SAFETY WATCHDOG ---
            time_since_last_packet = time.time() - last_packet_time
            if time_since_last_packet > cfg.max_safety_lag:
                print(f"SAFETY TRIGGER: Signal lost for {time_since_last_packet:.2f}s")
                action_queue.clear() # Dump queue
                # Robot will naturally stop if we don't send a command
            
            # --- D. EXECUTE ---
            if action_queue:
                # Pop the next immediate move
                next_action = action_queue.pop(0)
                
                # Send to motors
                # In current LeRobot, actions are dictionaries mapping motor names to values
                action_dict = {
                    key: next_action[i].item() 
                    for i, key in enumerate(robot.action_features)
                }
                robot.send_action(action_dict)
                
                if cfg.use_rerun:
                    rr.log("action", rr.Tensor(next_action))

            # --- E. VISUALIZE ---
            if cfg.use_rerun and "observation.images.laptop" in observation:
                # Log camera feed
                img_tensor = observation["observation.images.laptop"]
                # Convert CxHxW (Torch) -> HxWxC (Numpy) for Rerun
                img_np = img_tensor.permute(1, 2, 0).numpy()
                rr.log("camera/laptop", rr.Image(img_np))

            # --- F. TIMING ---
            # Maintain 30Hz
            elapsed = time.time() - loop_start
            sleep_time = max(0, dt - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nManual Stop.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("Bot Shutting Down...")
        robot.disconnect()
        print("Disconnected.")

if __name__ == "__main__":
    main()