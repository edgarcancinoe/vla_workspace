import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import yaml

ROOT_DIR = Path(__file__).resolve().parents[3]
src_root = ROOT_DIR / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

lerobot_src = ROOT_DIR.parent / "repos" / "lerobot" / "src"
if lerobot_src.exists() and str(lerobot_src) not in sys.path:
    sys.path.insert(0, str(lerobot_src))

from thesis_vla.robot.so101_control import SO101Control
from thesis_vla.vision import camera_calibration

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.teleoperators.so_leader.so_leader import SOLeader
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "robot" / "robot_config_dual.yaml"
ROBOT_KEYS = ("robot_a", "robot_b")
LEADER_KEYS = ("leader_a", "leader_b")
CAMERA_RECONNECT_INTERVAL_STEPS = 150


def _require_fields(section_name: str, section_data: dict, fields: list[str]) -> None:
    for field in fields:
        if section_data.get(field) in (None, ""):
            raise ValueError(f"Missing required field '{section_name}.{field}' in config.")


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    for robot_key in ROBOT_KEYS:
        if robot_key not in data:
            raise ValueError(f"Missing required top-level section '{robot_key}' in config.")
        _require_fields(
            robot_key,
            data[robot_key],
            ["name", "port", "urdf_path", "calibration_dir", "home_pose"],
        )
    for leader_key in LEADER_KEYS:
        if leader_key not in data:
            raise ValueError(f"Missing required top-level section '{leader_key}' in config.")
        _require_fields(leader_key, data[leader_key], ["name", "port"])

    return data


def _build_camera_config(robot_cfg: dict) -> tuple[dict, dict]:
    cameras = robot_cfg.get("cameras", {})
    camera_config = {}
    rectify_map = {}
    for camera_name, camera_info in cameras.items():
        if "id" not in camera_info:
            raise ValueError(f"Camera '{camera_name}' is missing required field 'id'.")
        camera_config[camera_name] = OpenCVCameraConfig(
            index_or_path=camera_info["id"],
            width=camera_info.get("width", 640),
            height=camera_info.get("height", 480),
            fps=camera_info.get("fps", 30),
        )
        rectify_map[camera_name] = bool(camera_info.get("rectify", False))
    return camera_config, rectify_map


def _camera_is_readable(camera_id: int) -> bool:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        cap.release()
        return False

    ok = False
    for _ in range(3):
        ret, _ = cap.read()
        if ret:
            ok = True
            break
    cap.release()
    return ok


def _camera_matches_requested_mode(camera_id: int, width: int, height: int) -> tuple[bool, int, int]:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        cap.release()
        return False, -1, -1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return actual_w == width and actual_h == height, actual_w, actual_h


def _filter_unavailable_cameras(cameras_cfg: dict, robot_key: str, skip_missing_cameras: bool) -> dict:
    if not skip_missing_cameras:
        return cameras_cfg

    filtered = {}
    for cam_name, cam_info in cameras_cfg.items():
        cam_id = cam_info.get("id")
        if cam_id is None:
            print(f"Warning: {robot_key}.{cam_name} has no camera id and will be skipped.")
            continue
        if _camera_is_readable(int(cam_id)):
            target_w = int(cam_info.get("width", 640))
            target_h = int(cam_info.get("height", 480))
            mode_ok, actual_w, actual_h = _camera_matches_requested_mode(int(cam_id), target_w, target_h)
            if mode_ok:
                filtered[cam_name] = cam_info
            else:
                print(
                    f"Warning: {robot_key}.{cam_name} (OpenCVCamera({cam_id})) does not support "
                    f"requested {target_w}x{target_h} (actual {actual_w}x{actual_h}). Skipping this camera."
                )
        else:
            print(
                f"Warning: {robot_key}.{cam_name} (OpenCVCamera({cam_id})) is unavailable. "
                "Skipping this camera."
            )

    if not filtered:
        print(f"Warning: No working cameras left for {robot_key}. Running without cameras.")

    return filtered


def _namespace_payload(namespace: str, payload: dict) -> dict:
    return {f"{namespace}/{k}": v for k, v in payload.items()}


def _print_pose(tag: str, control: SO101Control, observation: dict, step: int) -> None:
    obs_motor_vals = control.extract_motor_vals(observation)
    deg_vals = control.motor_to_deg(obs_motor_vals)
    pos, euler = control.fk_pose(obs_motor_vals)

    print(f"\n--- {tag} | Step {step} ---")
    print("Joint Positions (Motor Units | Degrees):")
    for i, joint_name in enumerate(control.JOINT_NAMES):
        m_val = obs_motor_vals[i]
        d_val = deg_vals[i]
        print(f"  {joint_name:15s}: {m_val:8.4f} units | {d_val:8.2f}°")

    print("\nEnd-Effector Pose (Cartesian):")
    print(f"  X: {pos[0]:8.4f} m")
    print(f"  Y: {pos[1]:8.4f} m")
    print(f"  Z: {pos[2]:8.4f} m")
    print(f"  Orientation (Euler xyz): Roll={euler[0]:.2f}°, Pitch={euler[1]:.2f}°, Yaw={euler[2]:.2f}°")


def _safe_disconnect(name: str, device) -> None:
    try:
        device.disconnect()
        print(f"Disconnected {name}.")
    except Exception as exc:
        print(f"Warning: Failed to disconnect {name}: {exc}")


def _disable_robot_cameras(robot, state: dict, robot_key: str, reason: Exception | str) -> None:
    reason_text = str(reason)
    print(f"Warning: {robot_key} camera stream failed ({reason_text}). Disabling cameras and continuing teleoperation.")

    for cam_name, cam in list(robot.cameras.items()):
        try:
            if getattr(cam, "is_connected", False):
                cam.disconnect()
        except Exception:
            pass
        finally:
            robot.cameras.pop(cam_name, None)

    state["rectify_map"] = {}


def _try_restore_robot_cameras(robot, state: dict, robot_key: str, step: int) -> None:
    if not state.get("cameras_disabled", False):
        return

    next_step = int(state.get("next_reconnect_step", 0))
    if step < next_step:
        return
    state["next_reconnect_step"] = step + CAMERA_RECONNECT_INTERVAL_STEPS

    recovered = []
    for cam_name, cam_info in state.get("camera_specs", {}).items():
        if cam_name in robot.cameras:
            continue

        cam_id = int(cam_info["id"])
        width = int(cam_info.get("width", 640))
        height = int(cam_info.get("height", 480))
        fps = int(cam_info.get("fps", 30))

        if not _camera_is_readable(cam_id):
            continue

        mode_ok, _, _ = _camera_matches_requested_mode(cam_id, width, height)
        if not mode_ok:
            continue

        cam = OpenCVCamera(
            OpenCVCameraConfig(
                index_or_path=cam_id,
                width=width,
                height=height,
                fps=fps,
            )
        )
        try:
            cam.connect()
            robot.cameras[cam_name] = cam
            state["rectify_map"][cam_name] = bool(state["rectify_map_template"].get(cam_name, False))
            recovered.append(cam_name)
        except Exception:
            try:
                cam.disconnect()
            except Exception:
                pass

    if recovered:
        state["cameras_disabled"] = False
        print(f"Info: {robot_key} cameras recovered: {', '.join(recovered)}")


def _get_observation_with_camera_fallback(robot, state: dict, robot_key: str, step: int) -> dict:
    _try_restore_robot_cameras(robot, state, robot_key, step)
    try:
        return robot.get_observation()
    except Exception as exc:
        if not state.get("cameras_disabled", False):
            _disable_robot_cameras(robot, state, robot_key, exc)
            state["cameras_disabled"] = True
            state["next_reconnect_step"] = step + CAMERA_RECONNECT_INTERVAL_STEPS
            return robot.get_observation()
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to dual-robot config yaml.",
    )
    calib_group = parser.add_mutually_exclusive_group()
    calib_group.add_argument(
        "--calibrate",
        action="store_true",
        help="Reset all calibration files and run full recalibration flow.",
    )
    calib_group.add_argument(
        "--calibrate-safe",
        action="store_true",
        help="Run calibration flow without deleting existing calibration files.",
    )
    parser.add_argument(
        "--strict-cameras",
        action="store_true",
        help="Fail fast if any configured camera is unavailable.",
    )
    args = parser.parse_args()

    try:
        config_data = _load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading config: {exc}")
        sys.exit(1)

    robot_states = {}
    for robot_key, leader_key in zip(ROBOT_KEYS, LEADER_KEYS):
        robot_cfg = dict(config_data[robot_key])
        leader_cfg = config_data[leader_key]

        robot_cfg["cameras"] = _filter_unavailable_cameras(
            robot_cfg.get("cameras", {}),
            robot_key=robot_key,
            skip_missing_cameras=not args.strict_cameras,
        )
        camera_config, rectify_map = _build_camera_config(robot_cfg)
        calibration_dir = Path(robot_cfg["calibration_dir"])
        wrist_roll_offset_deg = float(robot_cfg.get("wrist_roll_offset", 0.0))

        if args.calibrate:
            print(f"Force calibration for {robot_key}. Removing: {calibration_dir}")
            if calibration_dir.exists():
                shutil.rmtree(calibration_dir)
                print(f"Calibration files removed for {robot_key}.")

        follower_config = SO100FollowerConfig(
            port=robot_cfg["port"],
            id=robot_cfg["name"],
            cameras=camera_config,
            calibration_dir=calibration_dir,
        )
        teleop_config = SOLeaderTeleopConfig(
            port=leader_cfg["port"],
            id=leader_cfg["name"],
            calibration_dir=calibration_dir,
        )

        control = SO101Control(
            urdf_path=robot_cfg["urdf_path"],
            wrist_roll_offset=wrist_roll_offset_deg,
            home_pose=robot_cfg["home_pose"],
        )

        robot_states[robot_key] = {
            "robot": SO100Follower(follower_config),
            "leader": SOLeader(teleop_config),
            "control": control,
            "rectify_map": rectify_map,
            "rectify_map_template": dict(rectify_map),
            "namespace": robot_key,
            "cameras_disabled": False,
            "camera_specs": dict(robot_cfg.get("cameras", {})),
            "next_reconnect_step": 0,
        }

        print(f"{robot_key} follower port: {robot_cfg['port']}")
        print(f"{leader_key} leader port:   {leader_cfg['port']}")

    print("Connecting all devices...")
    try:
        for state in robot_states.values():
            do_calibrate = args.calibrate or args.calibrate_safe
            state["robot"].connect(calibrate=do_calibrate)
            state["leader"].connect(calibrate=do_calibrate)
    except Exception as exc:
        print(f"Error while connecting devices: {exc}")
        for key, state in robot_states.items():
            _safe_disconnect(f"{key} robot", state["robot"])
            _safe_disconnect(f"{key} leader", state["leader"])
        raise

    init_rerun(session_name="teleoperate_dual_debug")
    rr.log("world/origin", rr.Transform3D(translation=[0, 0, 0], mat3x3=np.eye(3)), static=True)
    print("Connected! Dual teleoperation is running.")

    step = 0
    print_interval = 100000000

    try:
        while True:
            for key, state in robot_states.items():
                observation = _get_observation_with_camera_fallback(state["robot"], state, key, step)
                action = state["leader"].get_action()
                control = state["control"]

                motor_vals = np.array([action[f"{joint_name}.pos"] for joint_name in control.JOINT_NAMES])
                motor_vals = control.apply_wrist_roll_offset(motor_vals)
                for i, joint_name in enumerate(control.JOINT_NAMES):
                    action[f"{joint_name}.pos"] = float(motor_vals[i])

                state["robot"].send_action(action)

                for cam_name, should_rectify in state["rectify_map"].items():
                    if should_rectify and cam_name in observation:
                        observation[cam_name] = camera_calibration.rectify_image(observation[cam_name], cam_name)

                if step % print_interval == 0:
                    _print_pose(key, control, observation, step)

                log_rerun_data(
                    observation=_namespace_payload(state["namespace"], observation),
                    action=_namespace_payload(state["namespace"], action),
                )

            rr.set_time("step", sequence=step)
            step += 1

    except KeyboardInterrupt:
        print("\nStopping dual teleoperation...")
    finally:
        for key, state in robot_states.items():
            _safe_disconnect(f"{key} robot", state["robot"])
            _safe_disconnect(f"{key} leader", state["leader"])


if __name__ == "__main__":
    main()
