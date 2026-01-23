
import time
import threading
import argparse
import yaml
import torch
import draccus
from pathlib import Path
from dataclasses import replace

# LeRobot Imports
from lerobot.async_inference.robot_client import RobotClient, RobotClientConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
LAUNCH_CONFIG_PATH = Path("launch_client.yaml")
ROBOT_CONFIG_PATH = Path("robot_config.yaml")

DATASET_REPO_ID = "edgarcancinoe/eval_baseline_tunnel"
FPS = 30

# -----------------------------------------------------------------------------
# CUSTOM CLIENT FOR RECORDING
# -----------------------------------------------------------------------------
class RecordingRobotClient(RobotClient):
    """
    Subclass of RobotClient that saves observations and actions to a LeRobotDataset
    while running the control loop.
    """
    def __init__(self, config: RobotClientConfig, dataset: LeRobotDataset):
        super().__init__(config)
        self.dataset = dataset

    def control_loop(self, task: str, verbose: bool = False):
        """Combined function for executing actions, streaming observations, and RECORDING data."""
        self.start_barrier.wait()
        self.logger.info("🔴 Control loop thread starting (WITH RECORDING ENABLED)")

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()
            
            # 1. Execute Action (if available from server)
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)

            # 2. Capture Observation & Stream to Server
            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(task, verbose)

            # 3. SAVE TO DATASET
            # We record every frame where we captured an observation.
            # If an action was performed recently, we associate it. 
            # Note: In async teleop, action might be None initially or repeated.
            # Ideally, we record the *executed state* of the robot. 
            # Since 'control_loop_action' returns values sent to motors, we use that.
            
            if _captured_observation is not None:
                # Prepare frame
                frame = {**_captured_observation}
                
                if _performed_action is not None:
                    # _performed_action is usually a Tensor. Convert to numpy/list.
                    if isinstance(_performed_action, torch.Tensor):
                         # Move to CPU if needed (though our client patch ensures CPU recv)
                        frame["action"] = _performed_action.cpu().numpy()
                    else:
                        frame["action"] = _performed_action
                
                # Only add if we have dataset features initialized
                # (RobotClient initializes robot connection, so we can setup dataset after connect)
                try:
                    self.dataset.add_frame(frame)
                except Exception as e:
                    self.logger.warning(f"Failed to save frame: {e}")

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print(f"🚀 Initializing Evaluation Client...")
    print(f"   Launch Config: {LAUNCH_CONFIG_PATH}")
    print(f"   Robot Config:  {ROBOT_CONFIG_PATH}")

    # 1. Load Client Configuration
    if not LAUNCH_CONFIG_PATH.exists():
        raise FileNotFoundError(f"{LAUNCH_CONFIG_PATH} not found!")
    
    # helper: load yaml dict
    with open(LAUNCH_CONFIG_PATH, 'r') as f:
        launch_cfg_dict = yaml.safe_load(f)

    # 2. Patch with Robot Config (Ports)
    if ROBOT_CONFIG_PATH.exists():
        with open(ROBOT_CONFIG_PATH, 'r') as f:
            robot_cfg_dict = yaml.safe_load(f)
        
        # Override port
        if "robot" in robot_cfg_dict and "port" in robot_cfg_dict["robot"]:
            print(f"   Overriding Robot Port: {robot_cfg_dict['robot']['port']}")
            # We need to inject this into the draccus config somehow, or just modify the dict before parsing if possible.
            # Since draccus parses files directly, we might need to pass args manually or use a trick.
            # Trick: Parse first, then modify object.
            pass
        
        follower_port = robot_cfg_dict["robot"]["port"]
    else:
        follower_port = None

    # 3. Parse Config into Object
    cfg = draccus.parse(RobotClientConfig, config_path=str(LAUNCH_CONFIG_PATH))
    
    # MANUAL OVERRIDE of Port
    if follower_port:
        cfg.robot.port = follower_port
        
    # FORCE BASELINE MODEL
    cfg.pretrained_name_or_path = "lerobot/smolvla_base"
    cfg.policy_type = "smolvla"
    print(f"   Forcing Baseline Model: {cfg.pretrained_name_or_path}")
        
    # Override Repo ID if needed
    # cfg.dataset_repo_id = DATASET_REPO_ID 

    print("✅ Configuration Loaded.")

    # 4. Initialize Client (This Connects the Robot!)
    # We must instantiate our custom class
    # We pass 'dataset=None' first, initialize it, then assign it.
    # Actually, we need robot features to init dataset.
    # RobotClient connects in .start() or __init__? 
    # RobotClient connects in __init__.
    
    print("🔌 Connecting to Robot & Server...")
    # Hack: We can't access robot.features until client is created.
    # But we can't create client fully without dataset if we put it in __init__
    # So we used a placeholder in __init__ or setters.
    
    # Helper to get features without connecting twice?
    # No, let's just let client connect.
    
    # Temporary dataset placeholder
    client = RecordingRobotClient(cfg, dataset=None)
    
    # 5. Initialize Dataset (Now that robot is connected)
    print("💾 Initializing Dataset...")
    features = hw_to_dataset_features(client.robot.features)
    
    dataset = LeRobotDataset.create(
        repo_id=DATASET_REPO_ID,
        fps=FPS,
        root=Path("data"),
        features=features,
        robot_type=client.robot.calibration_type,
    )
    
    client.dataset = dataset # Assign real dataset
    
    # 6. Run Loop
    print(f"🎬 Starting Recording Loop! (Task: {cfg.task})")
    print("   Press Ctrl+C to stop.")
    
    try:
        if client.start():
            # Start Action Receiver Thread
            action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
            action_receiver_thread.start()
            
            try:
                # Main Loop
                client.control_loop(task=cfg.task, verbose=True)
            except KeyboardInterrupt:
                print("\n⏹️ Stopping...")
            finally:
                client.stop()
                action_receiver_thread.join()
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Save Dataset
        print("💾 Saving Dataset...")
        dataset.consolidate()
        print("✅ Done!")

if __name__ == "__main__":
    main()