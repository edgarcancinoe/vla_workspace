


import time
import threading
import yaml
import torch
from pathlib import Path

# LeRobot Imports
from lerobot.async_inference.robot_client import RobotClient, RobotClientConfig

# =============================================================================
# CONFIGURATION
# =============================================================================
LAUNCH_CONFIG_PATH = Path("../config/launch_client.yaml")
ROBOT_CONFIG_PATH = Path("../config/robot_config.yaml")

# Model Configuration
MODEL_PATH = "edgarcancinoe/smolvla_finetuned_pkandplc20k"  # Baseline model
POLICY_TYPE = "smolvla"

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
        """Override to debug observation sending."""
        self.observation_count += 1
        
        if self.observation_count % 30 == 1:  # Print every 30 observations (~1 second)
            print(f"\n[OBS #{self.observation_count}] Capturing and sending observation to policy server...")
        
        result = super().control_loop_observation(task, verbose)
        
        if self.observation_count % 30 == 1:
            print(f"[OBS #{self.observation_count}] Observation sent successfully")
        
        return result
    
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
    
    # FORCE BASELINE MODEL
    cfg.pretrained_name_or_path = MODEL_PATH
    cfg.policy_type = POLICY_TYPE
    print(f"   Forcing Baseline Model: {cfg.pretrained_name_or_path}")
        
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