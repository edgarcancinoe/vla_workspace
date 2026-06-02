import logging
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING

import draccus
import yaml

from lerobot.utils.utils import log_say as lerobot_log_say

from thesis_vla.common.paths import DATASETS_OUTPUT_DIR, ROBOT_CONFIG_PATH
from thesis_vla.inference.eval_runtime import (
    DatasetPushConfig,
    ExecutionSmoothingConfig,
    init_eval_keyboard_controls,
    init_meshcat,
    push_eval_dataset_to_hub,
    recover_missing_eval_videos,
    set_custom_get_observation,
    set_custom_send_action,
    validate_lerobot_dataset_on_disk,
)

if TYPE_CHECKING:
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

try:
    from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
    from lerobot.async_inference.robot_client import RobotClient
    from lerobot.transport import services_pb2  # type: ignore
except Exception as exc:  # pragma: no cover - runtime environment issue, not logic
    RemotePolicyConfig = None
    TimedObservation = None
    RobotClient = object
    services_pb2 = None
    _RESIDENT_EVAL_IMPORT_ERROR = exc
else:
    _RESIDENT_EVAL_IMPORT_ERROR = None


@dataclass
class ResidentEvalConfig:
    policy_type: str = "xvla"
    pretrained_name_or_path: str = ""
    server_address: str = "127.0.0.1:8080"
    policy_device: str = "cpu"
    client_device: str = "cpu"
    actions_per_chunk: int | None = None
    chunk_size_threshold: float = 0.5
    task: str = ""
    task_prompts: list[str] = field(default_factory=list)
    num_episodes: int | None = None
    control_fps: float = 7.5
    camera_fps: int = 30
    episode_time_s: float = 90.0
    home_duration_s: float = 5.0
    home_fps: float | None = None
    dry_run: bool = False
    use_rerun: bool = False
    use_meshcat_viz: bool = False
    hf_user: str = "edgarcancinoe"
    eval_dataset_name: str = ""
    dataset_root: str = str(DATASETS_OUTPUT_DIR)
    start_from_scratch: bool = True
    resume_dataset: bool = False
    overwrite_dataset: bool = True
    auto_push_to_hub: bool = False
    use_voice: bool = True
    enable_ema_smoothing: bool = False
    ema_alpha_motor: float = 0.35
    ema_alpha_gripper: float = 0.25
    ema_reset_on_episode_start: bool = True
    enable_interpolation: bool = False
    interp_substeps: int = 3
    interp_min_delta_motor: float = 0.0
    interp_apply_to_gripper: bool = False
    interp_keep_control_fps_budget: bool = True
    interp_min_sleep_s: float = 0.0005
    hf_upload_max_retries: int = 3
    hf_upload_retry_backoff_s: float = 3.0


def resolve_episode_prompts(task: str, task_prompts: list[str], num_episodes: int | None) -> list[str]:
    prompts = [prompt.strip() for prompt in task_prompts if prompt and prompt.strip()] or ([task.strip()] if task and task.strip() else [])
    if not prompts:
        raise ValueError("Provide either task or at least one task_prompts entry.")
    total = len(prompts) if num_episodes is None else int(num_episodes)
    if total <= 0:
        raise ValueError(f"num_episodes must be positive, got {total}")
    return [prompts[i % len(prompts)] for i in range(total)]


def infer_actions_per_chunk(explicit_value: int | None, pretrained_name_or_path: str) -> int:
    if explicit_value is not None:
        return int(explicit_value)
    policy_cfg = PreTrainedConfig.from_pretrained(pretrained_name_or_path)
    inferred = getattr(policy_cfg, "n_action_steps", None) or getattr(policy_cfg, "chunk_size", None)
    if inferred is None:
        raise ValueError("Unable to infer actions_per_chunk from checkpoint config. Set actions_per_chunk explicitly.")
    return int(inferred)


def resolve_dataset_name(eval_dataset_name: str, pretrained_name_or_path: str) -> str:
    return eval_dataset_name or f"eval_remote_{pretrained_name_or_path.split('/')[-1]}"


def create_or_resume_dataset(repo_id: str, data_dir: Path, dataset_features: dict, robot_type: str, fps: float, resume_dataset: bool):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if resume_dataset:
        print(f"[x] Resuming dataset: {repo_id} at {data_dir}")
        dataset = LeRobotDataset(repo_id=repo_id, root=data_dir)
        return dataset, dataset.num_episodes
    print(f"[x] Creating new evaluation dataset: {repo_id}")
    dataset = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=dataset_features, robot_type=robot_type, use_videos=True, image_writer_threads=0, root=data_dir)
    return dataset, 0


class ResidentEvalRobotClient(RobotClient):
    def __init__(self, config: RobotClientConfig):
        if _RESIDENT_EVAL_IMPORT_ERROR is not None:
            raise RuntimeError("Resident eval runtime dependencies are unavailable.") from _RESIDENT_EVAL_IMPORT_ERROR
        super().__init__(config)
        self.action_names: list[str] = []
        self._action_receiver_thread: threading.Thread | None = None

    def configure_runtime(self, *, lerobot_features: dict, action_names: list[str], rename_map: dict[str, str]) -> None:
        self.action_names = list(action_names)
        self.policy_config = RemotePolicyConfig(
            self.config.policy_type,
            self.config.pretrained_name_or_path,
            lerobot_features,
            self.config.actions_per_chunk,
            self.config.policy_device,
            rename_map,
        )

    def _action_tensor_to_action_dict(self, action_tensor):
        action_tensor = action_tensor.squeeze(0).to("cpu")
        if self.action_names:
            return {key: float(action_tensor[i]) for i, key in enumerate(self.action_names)}
        return super()._action_tensor_to_action_dict(action_tensor)

    def begin_receiving(self, verbose: bool = False) -> None:
        if self._action_receiver_thread is not None:
            return
        self._action_receiver_thread = threading.Thread(target=self.receive_actions, kwargs={"verbose": verbose}, daemon=True)
        self._action_receiver_thread.start()
        self.start_barrier.wait()

    def reset_remote_session(self) -> None:
        self.stub.Ready(services_pb2.Empty())
        self.fps_tracker.reset()
        with self.action_queue_lock:
            while not self.action_queue.empty():
                self.action_queue.get_nowait()
        with self.latest_action_lock:
            self.latest_action = -1
        self.action_chunk_size = -1
        self.must_go.set()

    def step(self, task: str, verbose: bool = False):
        raw_observation = self.robot.get_observation()
        raw_observation["task"] = task
        if self._ready_to_send_observation():
            with self.latest_action_lock:
                latest_action = self.latest_action
            timed_observation = TimedObservation(timestamp=time.time(), observation=raw_observation, timestep=max(latest_action, 0))
            with self.action_queue_lock:
                timed_observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                queue_size = self.action_queue.qsize()
            self.send_observation(timed_observation)
            if timed_observation.must_go:
                self.must_go.clear()
            if verbose:
                self.logger.debug(f"Sent observation at queue size={queue_size} must_go={timed_observation.must_go}")
        action_tensor = None
        action_dict = None
        if self.actions_available():
            with self.action_queue_lock:
                self.action_queue_size.append(self.action_queue.qsize())
                timed_action = self.action_queue.get_nowait()
            action_tensor = timed_action.get_action()
            action_dict = self._action_tensor_to_action_dict(action_tensor)
            self.robot.send_action(action_dict)
            with self.latest_action_lock:
                self.latest_action = timed_action.get_timestep()
        return raw_observation, action_tensor, action_dict

    def stop(self):
        super().stop()
        if self._action_receiver_thread is not None:
            self._action_receiver_thread.join(timeout=1.0)
            self._action_receiver_thread = None


def run_remote_record_loop(*, client: ResidentEvalRobotClient, dataset: LeRobotDataset, task: str, fps: float, control_time_s: float, events: dict, use_rerun: bool = False):
    from lerobot.datasets.utils import build_dataset_frame
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.visualization_utils import log_rerun_data

    start_episode_t = time.perf_counter()
    timestamp = 0.0
    while timestamp < control_time_s and not events["exit_early"] and not events["stop_recording"]:
        start_loop_t = time.perf_counter()
        raw_observation, action_tensor, action_dict = client.step(task)
        if action_dict is not None:
            observation_frame = build_dataset_frame(dataset.features, raw_observation, prefix="observation")
            action_frame = build_dataset_frame(dataset.features, action_dict, prefix="action")
            dataset.add_frame({**observation_frame, **action_frame, "task": task})
            if use_rerun:
                log_rerun_data(observation=raw_observation, action=action_dict, compress_images=False)
        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(max(1 / fps - dt_s, 0.0))
        timestamp = time.perf_counter() - start_episode_t
    elapsed = time.perf_counter() - start_episode_t
    print(f"[remote_record_loop RETURN] elapsed={elapsed:.3f}s flags(exit_early={events.get('exit_early')}, rerecord={events.get('rerecord_episode')}, stop={events.get('stop_recording')})", flush=True)


@draccus.wrap()
def main(cfg: ResidentEvalConfig):
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.utils import hw_to_dataset_features
    from lerobot.policies.xvla.action_contract import get_so101_slice_spec
    from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
    from lerobot.utils.visualization_utils import init_rerun
    from thesis_vla.inference.xvla_runtime import resolve_xvla_rename_map
    from thesis_vla.robot.so101_control import SO101Control

    logging.info(pformat(cfg))
    if cfg.start_from_scratch and cfg.resume_dataset:
        raise ValueError("Cannot start from scratch and resume dataset at the same time.")
    prompts = resolve_episode_prompts(cfg.task, cfg.task_prompts, cfg.num_episodes)
    dataset_name = resolve_dataset_name(cfg.eval_dataset_name, cfg.pretrained_name_or_path)
    data_dir = Path(cfg.dataset_root) / dataset_name
    if cfg.start_from_scratch or cfg.overwrite_dataset:
        shutil.rmtree(data_dir, ignore_errors=True)
    with open(ROBOT_CONFIG_PATH, "r") as f:
        robot_cfg_data = yaml.safe_load(f)

    home_pose = robot_cfg_data["robot"].get("home_pose", {})
    so101 = SO101Control(urdf_path=robot_cfg_data["robot"]["urdf_path"], wrist_roll_offset=0.0, home_pose=home_pose)
    rectify_map = {name: info.get("rectify", False) for name, info in robot_cfg_data.get("cameras", {}).items()}
    cameras = {
        name: OpenCVCameraConfig(index_or_path=info["id"], width=info.get("width", 640), height=info.get("height", 480), fps=cfg.camera_fps)
        for name, info in robot_cfg_data.get("cameras", {}).items()
    }
    robot_config = SO100FollowerConfig(
        id=robot_cfg_data["robot"].get("name", "arm_follower"),
        cameras=cameras,
        port=robot_cfg_data["robot"]["port"],
        calibration_dir=Path(robot_cfg_data["robot"]["calibration_dir"]),
    )
    actions_per_chunk = infer_actions_per_chunk(cfg.actions_per_chunk, cfg.pretrained_name_or_path)
    client_cfg = RobotClientConfig(
        policy_type=cfg.policy_type,
        pretrained_name_or_path=cfg.pretrained_name_or_path,
        robot=robot_config,
        actions_per_chunk=actions_per_chunk,
        task=cfg.task,
        server_address=cfg.server_address,
        policy_device=cfg.policy_device,
        client_device=cfg.client_device,
        chunk_size_threshold=cfg.chunk_size_threshold,
        fps=cfg.control_fps,
        rename_map={},
    )
    client = ResidentEvalRobotClient(client_cfg)
    dataset = None
    keyboard_listener = None
    terminal_keyboard_listener = None
    should_attempt_push = False
    repo_id = f"{cfg.hf_user}/{dataset_name}"

    try:
        policy_cfg = PreTrainedConfig.from_pretrained(cfg.pretrained_name_or_path)
        include_eef_state = cfg.policy_type == "xvla" and getattr(policy_cfg, "action_mode", None) == "so101_ee6d"
        action_slice_spec = get_so101_slice_spec(getattr(policy_cfg, "action_mode", None)) if cfg.policy_type == "xvla" else None
        if cfg.policy_type == "xvla" and action_slice_spec is None:
            raise ValueError(f"Unsupported XVLA action_mode in checkpoint config: {getattr(policy_cfg, 'action_mode', None)!r}")
        if action_slice_spec is not None:
            mode_feature_spec = {"dtype": "float32", "shape": (action_slice_spec.real_dim,), "names": list(action_slice_spec.feature_names(suffix=".pos"))}
            action_features = {"action": mode_feature_spec}
            obs_features = hw_to_dataset_features(client.robot.observation_features, "observation")
            obs_features["observation.state"] = mode_feature_spec
        else:
            action_features = hw_to_dataset_features(client.robot.action_features, "action")
            obs_features = hw_to_dataset_features(client.robot.observation_features, "observation")
        dataset_features = {**action_features, **obs_features}
        dataset, episode_idx = create_or_resume_dataset(repo_id, data_dir, dataset_features, client.robot.name, cfg.control_fps, cfg.resume_dataset)
        active_xvla_rename_map = resolve_xvla_rename_map(dataset.meta.camera_keys) if cfg.policy_type == "xvla" else {}
        if cfg.policy_type == "xvla" and not active_xvla_rename_map:
            raise ValueError(f"Unable to resolve XVLA rename_map from dataset camera keys: {dataset.meta.camera_keys}")
        client.configure_runtime(lerobot_features=obs_features, action_names=dataset_features.get("action", {}).get("names", []), rename_map=active_xvla_rename_map)
        viz = init_meshcat(robot_cfg_data["robot"]["urdf_path"]) if cfg.use_meshcat_viz else None
        smoothing = ExecutionSmoothingConfig(
            enable_ema_smoothing=cfg.enable_ema_smoothing,
            ema_alpha_motor=cfg.ema_alpha_motor,
            ema_alpha_gripper=cfg.ema_alpha_gripper,
            ema_reset_on_episode_start=cfg.ema_reset_on_episode_start,
            enable_interpolation=cfg.enable_interpolation,
            interp_substeps=cfg.interp_substeps,
            interp_min_delta_motor=cfg.interp_min_delta_motor,
            interp_apply_to_gripper=cfg.interp_apply_to_gripper,
            interp_keep_control_fps_budget=cfg.interp_keep_control_fps_budget,
            interp_min_sleep_s=cfg.interp_min_sleep_s,
        )
        set_custom_get_observation(
            client.robot,
            so101=so101,
            rectify_map=rectify_map,
            active_xvla_rename_map=active_xvla_rename_map,
            eef_state_names=list(get_so101_slice_spec("so101_ee6d").names),
            include_eef_state=include_eef_state,
            dry_run=cfg.dry_run,
        )
        set_custom_send_action(client.robot, so101=so101, control_fps=cfg.control_fps, smoothing=smoothing, viz=viz, dry_run=cfg.dry_run)
        if hasattr(client.robot, "_xvla_reset_exec_smoothing_state"):
            client.robot._xvla_reset_exec_smoothing_state()
        keyboard_listener, events, terminal_keyboard_listener = init_eval_keyboard_controls()
        if cfg.use_rerun:
            init_rerun(session_name=f"resident_eval_{uuid.uuid4().hex[:8]}")
        if not client.start():
            raise RuntimeError("Failed to connect to resident policy server.")
        client.begin_receiving()
        log_say = lambda text: lerobot_log_say(text, play_sounds=cfg.use_voice)
        while episode_idx < len(prompts) and not events["stop_recording"]:
            task = prompts[episode_idx]
            log_say("Moving to home pose...")
            so101.reset_to_home(client.robot, duration_s=cfg.home_duration_s, fps=cfg.home_fps or cfg.control_fps, viz=viz)
            if hasattr(client.robot, "_xvla_reset_exec_smoothing_state") and cfg.ema_reset_on_episode_start:
                client.robot._xvla_reset_exec_smoothing_state(so101.read_motor_real(client.robot))
            client.reset_remote_session()
            log_say(f"Running resident eval episode {episode_idx + 1} of {len(prompts)}")
            print(f"[prompt] Episode {episode_idx + 1}: {task}")
            run_remote_record_loop(client=client, dataset=dataset, task=task, fps=cfg.control_fps, control_time_s=cfg.episode_time_s, events=events, use_rerun=cfg.use_rerun)
            if events["rerecord_episode"]:
                log_say("Re-recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue
            try:
                if len(dataset.episode_buffer) > 0:
                    dataset.save_episode(parallel_encoding=False)
                    episode_idx += 1
                else:
                    print("No frames recorded in episode buffer. Skipping save.")
            except Exception as save_error:
                print(f"[save] Failed to save episode {episode_idx}: {save_error}")
                dataset.clear_episode_buffer()
                raise
            events["exit_early"] = False
        should_attempt_push = cfg.auto_push_to_hub and not cfg.dry_run
    except Exception as e:
        print(f"[fatal] Resident eval failed: {e}")
        should_attempt_push = False
        raise
    finally:
        if terminal_keyboard_listener is not None:
            terminal_keyboard_listener.stop()
        if keyboard_listener is not None:
            keyboard_listener.stop()
        try:
            client.stop()
        except Exception:
            pass
        if dataset is not None:
            try:
                recover_missing_eval_videos(dataset)
                dataset.finalize()
                validate_lerobot_dataset_on_disk(dataset.root)
            except Exception as finalize_error:
                print(f"[finalize] Dataset finalization failed: {finalize_error}")
                should_attempt_push = False
                raise
        if should_attempt_push and dataset is not None:
            push_eval_dataset_to_hub(
                data_dir=dataset.root,
                repo_id=repo_id,
                push_config=DatasetPushConfig(max_retries=cfg.hf_upload_max_retries, retry_backoff_s=cfg.hf_upload_retry_backoff_s),
            )


if __name__ == "__main__":
    main()
