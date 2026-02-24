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

    # Attends discrepancy between URDF and real robot (used for KINEMATICS)
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
    # Use polarities defaults to TRUE, as they are necessary to use the official URDF file for IK 
    def motor_to_deg(self, motor_vals, use_polarities=True):
        """Converts motor units [0-100] to degrees."""
        motor_vals = np.asarray(motor_vals)
        # Precompute limit constants in degrees
        limits_deg = np.array([np.rad2deg(self.URDF_LIMITS_RAD[n]) for n in self.JOINT_NAMES])
        
        if use_polarities:
            polarities = np.array([self.POLARITIES[n] for n in self.JOINT_NAMES])
        else:
            polarities = 1.0
            
        # Handle both 1D (6,) and 2D (N, 6) arrays
        return (motor_vals / 100.0) * limits_deg * polarities

    def deg_to_motor(self, deg_vals, use_polarities=True):
        """Converts degrees to motor units [0-100]."""
        deg_vals = np.asarray(deg_vals)
        limits_deg = np.array([np.rad2deg(self.URDF_LIMITS_RAD[n]) for n in self.JOINT_NAMES])
        
        if use_polarities:
            polarities = np.array([self.POLARITIES[n] for n in self.JOINT_NAMES])
        else:
            polarities = 1.0
            
        val = (deg_vals / limits_deg) * 100.0 * polarities
        return np.clip(val, -100.0, 100.0)

    def motor_to_rad(self, motor_vals, use_polarities=True):
        return np.deg2rad(self.motor_to_deg(motor_vals, use_polarities=use_polarities))

    def rad_to_motor(self, rad_vals, use_polarities=True):
        return self.deg_to_motor(np.rad2deg(rad_vals), use_polarities=use_polarities)

    # ── Forward & Inverse Kinematics ──────────────────────────────────────────

    def fk(self, motor_vals):
        return self.kinematics.forward_kinematics(self.motor_to_deg(motor_vals))

    def fk_xyz(self, motor_vals):
        return self.fk(motor_vals)[:3, 3]

    def fk_xyz_chunk(self, motor_chunk, use_polarities=True):
        """Returns (N, 3) XYZ positions for a (N, 6) chunk of motor values."""
        degs = self.motor_to_deg(motor_chunk, use_polarities=use_polarities)
        return np.array([self.kinematics.forward_kinematics(d)[:3, 3] for d in degs])

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

    @staticmethod
    def rot6d_to_mat(r6d: np.ndarray) -> np.ndarray:
        """Recovers a 3×3 rotation matrix from a 6D rotation representation.

        The 6D vector stores the first two columns of the rotation matrix
        (r6d[0:3] = col0, r6d[3:6] = col1), which may be slightly non-orthonormal
        due to neural-network output noise.  A Gram-Schmidt step enforces
        orthonormality before the third column is recovered via the cross product.
        """
        a1 = np.asarray(r6d[:3], dtype=np.float64)
        a2 = np.asarray(r6d[3:6], dtype=np.float64)
        b1 = a1 / np.linalg.norm(a1)
        b2 = a2 - np.dot(b1, a2) * b1
        b2 = b2 / np.linalg.norm(b2)
        b3 = np.cross(b1, b2)
        return np.column_stack([b1, b2, b3])

    def ik_motor_6d(self, target_xyz, target_rot6d, seed_motor,
                    orientation_weight: float = 0.5) -> np.ndarray | None:
        """Solve IK for a full 6D EEF target (position + orientation).

        Converts the rot6d prediction to a 4×4 target pose via Gram-Schmidt
        orthonormalisation, then calls the IK solver with both position and
        orientation costs active.

        Args:
            target_xyz:         (3,) position target in metres.
            target_rot6d:       (6,) rotation target — first two columns of R.
            seed_motor:         (6,) current motor positions used as IK seed.
            orientation_weight: Weight for the orientation cost term relative to
                                position_weight=1.0.  Set to 0.0 to recover
                                position-only behaviour.  Default: 0.5.

        Returns:
            (6,) motor positions, or None if IK diverges / returns NaN.
        """
        R = self.rot6d_to_mat(target_rot6d)
        desired = np.eye(4)
        desired[:3, :3] = R
        desired[:3, 3] = target_xyz

        num_ik = len(self.active_joints)
        result_deg = self.kinematics.inverse_kinematics(
            current_joint_pos=self.motor_to_deg(seed_motor)[:num_ik],
            desired_ee_pose=desired,
            position_weight=1.0,
            orientation_weight=orientation_weight,
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

    def read_deg_real(self, robot, ignore_offset=False):
        """Reads joints in degrees, applying wrist roll offset."""
        motor_vals = self.read_motor_real(robot)
        deg_vals = self.motor_to_deg(motor_vals)
        if not ignore_offset and "wrist_roll" in self.JOINT_NAMES:
            idx = self.JOINT_NAMES.index("wrist_roll")
            deg_vals[idx] -= self.wrist_roll_offset
        return deg_vals

    def send_motor_real(self, robot, vals):
        """Sends motor values to robot [0-100 range]."""
        robot.send_action({f"{n}.pos": float(vals[i]) for i, n in enumerate(self.JOINT_NAMES)})

    def send_deg_real(self, robot, deg_vals, ignore_offset=False):
        """Sends joints in degrees, applying wrist roll offset."""
        cmd_deg = deg_vals.copy()
        if not ignore_offset and "wrist_roll" in self.JOINT_NAMES:
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

    def execute_joint_trajectory(self, robot, waypoints_deg, fps=30, ignore_offset=False):
        """Sends a sequence of joint waypoints at a fixed rate."""
        next_tick = time.perf_counter()
        for deg in waypoints_deg:
            self.send_deg_real(robot, deg, ignore_offset=ignore_offset)
            next_tick += 1.0 / fps
            precise_sleep(max(0.0, next_tick - time.perf_counter()))

    def execute_motor_trajectory(self, robot, waypoints_motor, fps=30):
        """Sends a sequence of motor units at a fixed rate."""
        next_tick = time.perf_counter()
        for motor in waypoints_motor:
            self.send_motor_real(robot, motor)
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

    def reset_to_home(self, robot, duration_s=3.0, fps=50, ignore_offset=False, viz=None):
        """Moves robot to target defined in self.home_pose over duration_s seconds using Joint Interpolation."""
        print(f"Moving to home pose over {duration_s:.1f}s...")
        steps = max(1, int(round(duration_s * fps)))
        
        if robot:
            start_deg = self.read_deg_real(robot, ignore_offset=ignore_offset)
        else:
            # If no robot, we might be in simulation. Use 0s or current viz state?
            # For simplicity, if robot is None, we assume we want to home the viz from its current state.
            # But SO101Control doesn't track current state besides robot/viz calls.
            # We'll default to starting from zeros if no robot.
            start_deg = np.zeros(len(self.JOINT_NAMES))
            
        goal_deg = np.array([float(self.home_pose.get(f"{n}.pos", 0.0)) for n in self.JOINT_NAMES])
        
        waypoints = self.interpolate_joint(start_deg, goal_deg, steps)
        
        # Execute the trajectory using normalized motor units
        next_tick = time.perf_counter()
        for deg in waypoints:
            cmd_deg = deg.copy()
            if not ignore_offset and "wrist_roll" in self.JOINT_NAMES:
                idx = self.JOINT_NAMES.index("wrist_roll")
                cmd_deg[idx] += self.wrist_roll_offset
            
            motor_vals = self.deg_to_motor(cmd_deg)
            self.send_and_display(motor_vals, robot, viz)
            
            next_tick += 1.0 / fps
            precise_sleep(max(0.0, next_tick - time.perf_counter()))
            
        print("Home pose reached.")

    def move_to_xyz(self, robot, target_xyz, ref_pose, seed_motor, duration_s=3.0, viz=None, fps=20, max_step=4.0):
        """Moves robot to XYZ target over duration_s seconds using Cartesian Interpolation."""
        steps = max(1, int(round(duration_s * fps)))
        current_xyz = self.fk_xyz(self.sync_state(seed_motor, robot))
        
        waypoints = self.interpolate_cartesian(current_xyz, target_xyz, steps)
        return self.execute_cartesian_trajectory(robot, waypoints, ref_pose, seed_motor, viz=viz, fps=fps, max_step=max_step)
