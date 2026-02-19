


import time
import threading
import yaml
import torch
from pathlib import Path

# LeRobot Imports
import sys
from pathlib import Path

# Add workspace root to sys.path to allow importing from scripts/utils
WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.append(str(WORKSPACE_ROOT))

# Now we can import from scripts/utils
from utils import camera_calibration

from lerobot.async_inference.robot_client import RobotClient, RobotClientConfig

# =============================================================================
# CONFIGURATION
# =============================================================================
# Use WORKSPACE_ROOT from imports section for robust paths
LAUNCH_CONFIG_PATH = WORKSPACE_ROOT / "config" / "launch_client.yaml"
ROBOT_CONFIG_PATH = WORKSPACE_ROOT / "config" / "robot_config.yaml"


# Denormalization Settings
ENABLE_DENORMALIZATION = False    # True for baseline model, False for fine-tuned
POLICY_OUTPUT_RANGE = (-1, 1)  # Range that baseline model outputs in
# Action ranges for SO-100 robot
# Robot uses MotorNormMode.RANGE_M100_100 which expects -100 to 100
ACTION_RANGES = {
    "shoulder_pan.pos": (-100, 100),
    "shoulder_lift.pos": (-100, 100),
    "elbow_flex.pos": (-100, 100),
    "wrist_flex.pos": (-100, 100),
    "wrist_roll.pos": (-100, 100),
    "gripper.pos": (0, 100),  # Gripper uses RANGE_0_100
}

# Load Rectification Flags and Dynamic Camera Mapping
CAMERA_NAME_MAPPING = {}
try:
    with open(ROBOT_CONFIG_PATH, "r") as f:
        _robot_config_data = yaml.safe_load(f)
    print(f"Loaded Robot Config for Rectification: {_robot_config_data.get('rectification')}")
    RECTIFY_TOP = _robot_config_data.get("rectification", {}).get("top", True)
    RECTIFY_WRIST = _robot_config_data.get("rectification", {}).get("wrist", False)

    # Build Map: cameraX -> index -> name (top/wrist)
    # 1. Get index->name from robot config
    _idx_to_name = {v: k for k, v in _robot_config_data.get('cameras', {}).items()}
    
    # 2. Get cameraX->index from launch config
    with open(LAUNCH_CONFIG_PATH, "r") as f:
        _launch_config_data = yaml.safe_load(f)
        
    if 'robot' in _launch_config_data and 'cameras' in _launch_config_data['robot']:
        for _cam_key, _cam_conf in _launch_config_data['robot']['cameras'].items():
            _idx = _cam_conf.get('index_or_path')
            if _idx in _idx_to_name:
                CAMERA_NAME_MAPPING[_cam_key] = _idx_to_name[_idx]
                
    print(f"Dynamic Camera Mapping: {CAMERA_NAME_MAPPING}")

except Exception as e:
    print(f"WARNING: Could not load config/mapping from {ROBOT_CONFIG_PATH}: {e}")
    RECTIFY_TOP = True
    RECTIFY_WRIST = True
    # Fallback to defaults if detection fails
    CAMERA_NAME_MAPPING = {"camera1": "wrist", "camera2": "top"}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def denormalize_actions(actions_dict, policy_range=(-1, 1)):
    """Convert policy outputs to robot action ranges."""
    p_min, p_max = policy_range
    denormalized = {}
    for key, value in actions_dict.items():
        if key in ACTION_RANGES:
            r_min, r_max = ACTION_RANGES[key]
            normalized = (value - p_min) / (p_max - p_min)
            denormalized[key] = r_min + normalized * (r_max - r_min)
        else:
            denormalized[key] = value
    return denormalized

# =============================================================================
# DEBUG CLIENT
# =============================================================================
class DebugRobotClient(RobotClient):
    """RobotClient with detailed pipeline debugging."""
    
    def __init__(self, config: RobotClientConfig):
        super().__init__(config)
        self.action_count = 0
        self.observation_count = 0
        self.received_action_chunks = 0
    
    def control_loop_observation(self, task: str, verbose: bool = False):
        """Override to debug observation sending and fix Rerun color visualization."""
        try:
            # 1. Capture observation (Raw Observation from Robot)
            # This logic mimics RobotClient.control_loop_observation but fixes the Rerun logging
            
            # Helper to access parent methods we can't easily super() call granularly
            # We will basically reimplement the coordination logic to inject our logging.
            
            import time
            import rerun as rr
            import numpy as np
            import torch
            from lerobot.async_inference.robot_client import TimedObservation
            
            # --- START COPY FROM PARENT ---
            start_time = time.perf_counter()

            raw_observation = self.robot.get_observation()
            
            # --- APPLY RECTIFICATION ---
            # Undistort images before any processing/sending
            # MAPPING: Derived dynamically from configs
            # e.g. {"camera1": "wrist", "camera2": "top"}
            key_to_calib = CAMERA_NAME_MAPPING
            
            for key, value in raw_observation.items():
                if isinstance(value, np.ndarray) and value.ndim == 3: # Image
                    calib_name = key_to_calib.get(key, key)
                    
                    # Apply flags
                    if calib_name == "top" and not RECTIFY_TOP:
                        continue
                    if calib_name == "wrist" and not RECTIFY_WRIST:
                        continue
                        
                    # key matches camera name in config (e.g. 'camera1', 'camera2')
                    # We pass 'calib_name' to ensure we use the correct matrix (wrist vs top)
                    raw_observation[key] = camera_calibration.rectify_image(value, calib_name)
            
            raw_observation["task"] = task

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )
            
            obs_capture_time = time.perf_counter() - start_time
            
            # Must-go logic
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            self.send_observation(observation)
            
            if observation.must_go:
                self.must_go.clear()

            # --- DEBUG LOGGING ---
            self.observation_count += 1
            if self.observation_count % 30 == 1:
                print(f"[OBS #{self.observation_count}] Sent observation (Queue: {current_queue_size})")

            # --- RERUN LOGGING (FIXED) ---
            if self.config.use_rerun:
                rr.set_time_nanos("log_time", int(observation.get_timestamp() * 1e9))
                
                for key, value in raw_observation.items():
                    if isinstance(value, np.ndarray) and value.ndim == 3: # Likely an image
                        # FIXED: Do NOT swap channels. OpenCVCamera returns RGB, Rerun expects RGB.
                        # Original RobotClient does cv2.cvtColor(value, cv2.COLOR_BGR2RGB) which caused the Blue Tint.
                        rr.log(f"camera/{key}", rr.Image(value))
                    elif isinstance(value, (float, int)) or (isinstance(value, torch.Tensor) and value.numel() == 1):
                        rr.log(f"state/{key}", rr.Scalars(value))
                        
            return raw_observation
            
        except Exception as e:
            self.logger.error(f"Error in debug observation sender: {e}")

    
    def control_loop_action(self, verbose: bool = False):
        """Override to apply denormalization BEFORE sending to robot."""
        import time
        import rerun as rr
        
        self.action_count += 1
        
        # Check queue status
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
        
        if queue_size == 0:
            if self.action_count % 100 == 1:
                print(f"[WARNING] [Action #{self.action_count}] Queue is EMPTY - no actions to execute")
            return None
        
        # Get action from queue (copied from parent)
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            timed_action = self.action_queue.get_nowait()
        get_end  = time.perf_counter() - get_start
        
        # Convert tensor to action dict
        action = self._action_tensor_to_action_dict(timed_action.get_action())
        
        # APPLY DENORMALIZATION HERE (before sending to robot)
        if ENABLE_DENORMALIZATION:
            if self.action_count % 20 == 1:
                print(f"\n{'='*60}")
                print(f"[Action #{self.action_count}] EXECUTING ACTION")
                print(f"{'='*60}")
                print(f"   Queue Size: {queue_size}")
                print(f"   [DENORM] Original (policy output -1 to 1):")
                for key, value in action.items():
                    if isinstance(value, (int, float)):
                        print(f"      {key}: {value:.4f}")
            
            # Apply denormalization
            action = denormalize_actions(action, POLICY_OUTPUT_RANGE)
            
            if self.action_count % 20 == 1:
                print(f"   [DENORM] Denormalized (robot expects -100 to 100):")
                for key, value in action.items():
                    if isinstance(value, (int, float)):
                        print(f"      {key}: {value:.4f}")
                print(f"{'='*60}\n")
        elif self.action_count % 20 == 1:
            print(f"\n{'='*60}")
            print(f"[Action #{self.action_count}] EXECUTING ACTION")
            print(f"{'='*60}")
            print(f"   Queue Size: {queue_size}")
            print(f"   Action Values:")
            for key, value in action.items():
                if isinstance(value, (int, float)):
                    print(f"      {key}: {value:.4f}")
            print(f"{'='*60}\n")
        
        # NOW send to robot with denormalized values
        performed_action = self.robot.send_action(action)
        
        # Update latest action tracking
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()
        
        # Logging (from parent)
        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()
            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )
            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )
        
        # Rerun logging (from parent)
        if self.config.use_rerun:
            rr.set_time_nanos("log_time", int(timed_action.get_timestamp() * 1e9))
            rr.log("action", rr.Tensor(timed_action.get_action()))
        
        return performed_action
    
    def receive_actions(self, verbose: bool = False):
        """Override to debug action receiving from server."""
        print("\n[RECEIVE THREAD] Starting to listen for actions from server...")
        print("[RECEIVE THREAD] Waiting for policy server to send actions...")
        
        # Wrap parent's receive_actions to intercept the first chunk
        original_aggregate = self._aggregate_action_queues
        first_chunk_detected = [False]  # Use list for closure mutability
        
        def wrapped_aggregate(timed_actions, aggregate_fn):
            if not first_chunk_detected[0] and len(timed_actions) > 0:
                first_chunk_detected[0] = True
                print("\n" + "="*60)
                print("[RECEIVE THREAD] FIRST ACTION CHUNK RECEIVED FROM POLICY!")
                print("="*60)
                print(f"   Number of actions in chunk: {len(timed_actions)}")
                print(f"   First action timestep: {timed_actions[0].get_timestep()}")
                print("="*60 + "\n")
            return original_aggregate(timed_actions, aggregate_fn)
        
        self._aggregate_action_queues = wrapped_aggregate
        super().receive_actions(verbose=True)

# =============================================================================
# MAIN
# -----------------------------------------------------------------------------
def main():
    print(f"Initializing Baseline Model Execution...")
    print(f"Launch Config: {LAUNCH_CONFIG_PATH}")
    print(f"Robot Config:  {ROBOT_CONFIG_PATH}")

    # 1. Load Client Configuration (Manual YAML load)
    if not LAUNCH_CONFIG_PATH.exists():
        raise FileNotFoundError(f"{LAUNCH_CONFIG_PATH} not found!")
    
    with open(LAUNCH_CONFIG_PATH, 'r') as f:
        launch_cfg_dict = yaml.safe_load(f)

    # 1.5 Dynamic Camera mapping based on Policy Type
    policy_type = launch_cfg_dict.get("policy_type", "smolvla")
    print(f"   Detected Policy Type: {policy_type}")

    if "robot" in launch_cfg_dict and "cameras" in launch_cfg_dict["robot"]:
        original_cameras = launch_cfg_dict["robot"]["cameras"]
        new_cameras = {}
        
        for cam_key, cam_conf in original_cameras.items():
            if isinstance(cam_conf, dict): # Ensure it's a dict (handle empty/None from YAML)
                idx = cam_conf.get("index_or_path")
                
                # Key Mapping Logic
                new_key = cam_key # Default to original
                
                if policy_type == "smolvla":
                        if idx == 1: new_key = "camera1"
                        elif idx == 0: new_key = "camera2"
                elif policy_type == "xvla":
                        if idx == 1: new_key = "image"
                        elif idx == 0: new_key = "image2"
                
                if new_key != cam_key:
                    print(f"   [Config] Remapping camera index {idx}: '{cam_key}' -> '{new_key}'")
                
                new_cameras[new_key] = cam_conf
            
        launch_cfg_dict["robot"]["cameras"] = new_cameras

    # 2. Patch with Robot Config (Ports)
    if ROBOT_CONFIG_PATH.exists():
        with open(ROBOT_CONFIG_PATH, 'r') as f:
            robot_cfg_dict = yaml.safe_load(f)
        
        # Override port is CRITICAL for draccus parsing
        if "robot" in robot_cfg_dict and "port" in robot_cfg_dict["robot"]:
            follower_port = robot_cfg_dict["robot"]["port"]
            print(f"   Injecting Robot Port: {follower_port}")
            # Ensure 'robot' dict exists in launch config
            if "robot" not in launch_cfg_dict:
                launch_cfg_dict["robot"] = {}
            launch_cfg_dict["robot"]["port"] = follower_port
            
    import draccus.parsers.decoding as draccus_decoding
    cfg = draccus_decoding.decode(RobotClientConfig, launch_cfg_dict)
        
    print("Configuration Loaded:")
    print(format(cfg))
    time.sleep(5)

    # 3. Initialize Client (This Connects the Robot!)
    print("Connecting to Robot & Server...")
    client = DebugRobotClient(cfg)
    
    # 3.5 VERIFY AND ENABLE TORQUE (Critical for movement!)
    print("\n" + "="*60)
    print("VERIFYING MOTOR TORQUE STATUS")
    print("="*60)
    
    try:
        # Read current torque enable status
        torque_status = client.robot.bus.sync_read("Torque_Enable")
        print("Current Torque Status:")
        for motor, status in torque_status.items():
            status_str = "[ENABLED]" if status else "[DISABLED]"
            print(f"   {motor}: {status_str}")
        
        # Enable torque on all motors
        print("\nEnabling torque on all motors...")
        client.robot.bus.enable_torque()
        
        # Verify it worked
        torque_status_after = client.robot.bus.sync_read("Torque_Enable")
        all_enabled = all(torque_status_after.values())
        
        if all_enabled:
            print("[SUCCESS] All motors torque enabled!")
        else:
            print("[WARNING] Some motors still have torque disabled:")
            for motor, status in torque_status_after.items():
                if not status:
                    print(f"   [DISABLED] {motor}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"[WARNING] Could not verify torque status: {e}")
        print("Proceeding anyway...\n")

    
    # 4. Run Loop
    print("Press Ctrl+C to stop.")
    
    try:
        if client.start():
            # Start Action Receiver Thread
            action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
            action_receiver_thread.start()
            
            try:
                # Main Control Loop - Execute Baseline Model
                client.control_loop(task=cfg.task, verbose=True)
            except KeyboardInterrupt:
                print("\nStopping...")
            finally:
                client.stop()
                action_receiver_thread.join()
                
    except Exception as e:
        print(f"Error: {e}")
    
    print("Done!")

if __name__ == "__main__":
    main()