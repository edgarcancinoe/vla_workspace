import time
import numpy as np
from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.robot_utils import precise_sleep

class SO101Control:
    """Wrapper for SO-ARM100 Kinematics model and motor unit conversions."""
    
    JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']

    URDF_LIMITS_RAD = {
        'shoulder_pan':  1.91986,
        'shoulder_lift': 1.74533,
        'elbow_flex':    1.69000,
        'wrist_flex':    1.65806,
        'wrist_roll':    2.84121,
        'gripper':       1.74533,
    }

    POLARITIES = {
        'shoulder_pan':   1.0,
        'shoulder_lift': -1.0,
        'elbow_flex':     1.0,
        'wrist_flex':     1.0,
        'wrist_roll':     1.0,
        'gripper':        1.0,
    }

    def __init__(self, urdf_path, target_frame_name="gripper_frame_link", active_joints=None, wrist_roll_offset=0.0, home_pose=None):
        self.urdf_path = urdf_path
        self.target_frame_name = target_frame_name
        self.active_joints = active_joints if active_joints is not None else self.JOINT_NAMES
        self.wrist_roll_offset = wrist_roll_offset
        self.home_pose = home_pose if home_pose is not None else {f"{n}.pos": 0.0 for n in self.JOINT_NAMES}
        self.kinematics = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name=target_frame_name,
            joint_names=self.active_joints,
        )
        self.trail_counter = 0

    # ── Unit Conversions ──────────────────────────────────────────────────────

    def motor_to_deg(self, motor_vals):
        length = min(len(motor_vals), len(self.JOINT_NAMES))
        out = np.zeros(length)
        for i in range(length):
            n = self.JOINT_NAMES[i]
            out[i] = (motor_vals[i] / 100.0) * np.rad2deg(self.URDF_LIMITS_RAD[n]) * self.POLARITIES[n]
        return out

    def deg_to_motor(self, deg_vals):
        length = min(len(deg_vals), len(self.JOINT_NAMES))
        out = np.zeros(length)
        for i in range(length):
            n = self.JOINT_NAMES[i]
            limit_deg = np.rad2deg(self.URDF_LIMITS_RAD[n])
            # Clip to [-100, 100] to prevent motor driver overload errors
            val = (deg_vals[i] / limit_deg) * 100.0 * self.POLARITIES[n]
            out[i] = np.clip(val, -100.0, 100.0)
        return out

    def motor_to_rad(self, motor_vals):
        return np.deg2rad(self.motor_to_deg(motor_vals))

    def rad_to_motor(self, rad_vals):
        return self.deg_to_motor(np.rad2deg(rad_vals))

    # ── Forward & Inverse Kinematics ──────────────────────────────────────────

    def fk(self, motor_vals):
        return self.kinematics.forward_kinematics(self.motor_to_deg(motor_vals))

    def fk_xyz(self, motor_vals):
        return self.fk(motor_vals)[:3, 3]

    def ik_motor(self, target_xyz, ref_pose_4x4, seed_motor):
        desired = ref_pose_4x4.copy()
        desired[:3, 3] = target_xyz
        
        num_ik = len(self.active_joints)
        
        result_deg = self.kinematics.inverse_kinematics(
            current_joint_pos=self.motor_to_deg(seed_motor)[:num_ik],
            desired_ee_pose=desired,
            position_weight=1.0,
            orientation_weight=0.0,
        )
        if np.any(np.isnan(result_deg)):
            return None
            
        full_result_deg = self.motor_to_deg(seed_motor)
        full_result_deg[:num_ik] = result_deg
        return self.deg_to_motor(full_result_deg)

    # ── Hardware IO Helpers ──────────────────────────────────────────────────

    def read_motor_real(self, robot):
        """Reads motor values from robot [0-100 range]."""
        obs = robot.get_observation()
        vals = np.zeros(len(self.JOINT_NAMES))
        for i, n in enumerate(self.JOINT_NAMES):
            vals[i] = float(obs.get(f"{n}.pos", 0.0))
        return vals

    def read_deg_real(self, robot):
        """Reads joints in degrees, applying wrist roll offset."""
        motor_vals = self.read_motor_real(robot)
        deg_vals = self.motor_to_deg(motor_vals)
        if "wrist_roll" in self.JOINT_NAMES:
            idx = self.JOINT_NAMES.index("wrist_roll")
            deg_vals[idx] -= self.wrist_roll_offset
        return deg_vals

    def send_motor_real(self, robot, vals):
        """Sends motor values to robot [0-100 range]."""
        robot.send_action({f"{n}.pos": float(vals[i]) for i, n in enumerate(self.JOINT_NAMES)})

    def send_deg_real(self, robot, deg_vals):
        """Sends joints in degrees, applying wrist roll offset."""
        cmd_deg = deg_vals.copy()
        if "wrist_roll" in self.JOINT_NAMES:
            idx = self.JOINT_NAMES.index("wrist_roll")
            cmd_deg[idx] += self.wrist_roll_offset
        motor_vals = self.deg_to_motor(cmd_deg)
        self.send_motor_real(robot, motor_vals)

    def robot_sleep(self, dt, robot=None):
        if robot:
            precise_sleep(dt)
        else:
            time.sleep(dt)

    def send_and_display(self, motor_vals, robot=None, viz=None):
        if robot:
            self.send_motor_real(robot, motor_vals)
        if viz:
            viz.display(self.motor_to_rad(motor_vals))

    def sync_state(self, current_motor, robot=None):
        return self.read_motor_real(robot) if robot else current_motor

    def move_to_motor(self, current, target, duration_s, robot=None, viz=None, dt=0.05):
        steps = max(1, int(round(duration_s / dt)))
        for i in range(1, steps + 1):
            val = current + (target - current) * (i / steps)
            self.send_and_display(val, robot, viz)
            self.robot_sleep(dt, robot)
        
        if robot:
            time.sleep(0.2)
            return self.read_motor_real(robot)
        return target

    # ── Interpolation Math (No hardware IO) ──────────────────────────────────
    
    def interpolate_joint(self, start_deg, goal_deg, steps):
        """Returns (steps, 6) array of joint waypoints in degrees."""
        waypoints = []
        for i in range(1, steps + 1):
            alpha = i / steps
            waypoints.append(start_deg + (goal_deg - start_deg) * alpha)
        return np.array(waypoints)

    def interpolate_cartesian(self, start_xyz, target_xyz, steps):
        """Returns (steps, 3) array of XYZ waypoints."""
        waypoints = []
        for i in range(1, steps + 1):
            t = i / steps
            waypoints.append((1 - t) * start_xyz + t * target_xyz)
        return np.array(waypoints)

    # ── Execution Loops (Hardware IO) ────────────────────────────────────────

    def execute_joint_trajectory(self, robot, waypoints_deg, fps=30):
        """Sends a sequence of joint waypoints at a fixed rate."""
        next_tick = time.perf_counter()
        for deg in waypoints_deg:
            self.send_deg_real(robot, deg)
            next_tick += 1.0 / fps
            precise_sleep(max(0.0, next_tick - time.perf_counter()))

    def execute_cartesian_trajectory(self, robot, waypoints_xyz, ref_pose, seed_motor, viz=None, fps=30, max_step=4.0):
        """Solves IK for each XYZ waypoint and sends commands."""
        current_motor = self.sync_state(seed_motor, robot)
        current_seed  = current_motor.copy()
        dt = 1.0 / fps

        for i, xyz in enumerate(waypoints_xyz):
            target_motor = self.ik_motor(xyz, ref_pose, current_seed)
            if target_motor is None:
                print(f"  IK failed at waypoint {i}, xyz={xyz.round(4)}")
                self.robot_sleep(dt, robot)
                continue

            delta = np.clip(target_motor - current_motor, -max_step, max_step)
            current_motor = current_motor + delta
            current_seed  = current_motor.copy()

            self.send_and_display(current_motor, robot, viz)

            # Optional Meshcat visualization
            if viz and hasattr(viz, 'viewer'):
                import meshcat.geometry as g
                import meshcat.transformations as tf
                actual_xyz = self.fk_xyz(current_motor)
                self.trail_counter += 1
                name = f"trail/pt_{self.trail_counter}"
                viz.viewer[name].set_object(g.Sphere(0.003), g.MeshLambertMaterial(color=0xaaaaaa))
                viz.viewer[name].set_transform(tf.translation_matrix(actual_xyz))

            self.robot_sleep(dt, robot)
        
        return self.sync_state(current_motor, robot)

    # ── High Level API ────────────────────────────────────────────────────────

    def reset_to_home(self, robot, duration_s=3.0, fps=50):
        """Moves robot to target defined in self.home_pose over duration_s seconds using Joint Interpolation."""
        print(f"Moving to home pose over {duration_s:.1f}s...")
        steps = max(1, int(round(duration_s * fps)))
        
        start_deg = self.read_deg_real(robot)
        goal_deg = np.array([float(self.home_pose.get(f"{n}.pos", 0.0)) for n in self.JOINT_NAMES])
        
        waypoints = self.interpolate_joint(start_deg, goal_deg, steps)
        self.execute_joint_trajectory(robot, waypoints, fps=fps)
        print("Home pose reached.")

    def move_to_xyz(self, robot, target_xyz, ref_pose, seed_motor, duration_s=3.0, viz=None, fps=20, max_step=4.0):
        """Moves robot to XYZ target over duration_s seconds using Cartesian Interpolation."""
        steps = max(1, int(round(duration_s * fps)))
        current_xyz = self.fk_xyz(self.sync_state(seed_motor, robot))
        
        waypoints = self.interpolate_cartesian(current_xyz, target_xyz, steps)
        return self.execute_cartesian_trajectory(robot, waypoints, ref_pose, seed_motor, viz=viz, fps=fps, max_step=max_step)
