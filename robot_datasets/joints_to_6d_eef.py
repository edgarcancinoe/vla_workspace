#!/usr/bin/env python3
"""
Script to create a new dataset from an existing SOARM101 dataset, converting
joint positions to 6D end-effector rotational representations expected by XVLA.
"""

import argparse
import os
import re
import shutil
import json
import sys
from pathlib import Path

import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq

from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.policies.xvla.utils import mat_to_rotate6d

WORKSPACE_ROOT = str(Path(__file__).resolve().parent.parent)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)
from robot_control.so101_control import SO101Control

# --- Configuration ---
# DATASET_ID = "edgarcancinoe/soarm101_pickplace_orange_050e_fw_open"
# OUT_DATASET_ID = "edgarcancinoe/soarm101_pickplace_6d"

DATASET_ID = "edgarcancinoe/soarm101_pickplace_orange_240e_fw_closed"
OUT_DATASET_ID = "edgarcancinoe/soarm101_pickplace_6d_240e_fw_closed"

URDF_PATH = "/Users/edgarcancino/Documents/Academic/EMAI Thesis/repos/SO-ARM100/Simulation/SO101/so101_new_calib.urdf" 
JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
PUSH_TO_HUB = "--push" in sys.argv  # Pushes dataset to huggingface hub
# ---------------------

def main():
    if not URDF_PATH:
        print("Please configure URDF_PATH at the top of the script.")
        return

    joint_names = JOINT_NAMES
    print(f"Using joint names: {joint_names}")

    print(f"Loading dataset: {DATASET_ID}")
    # Load with video downloading as requested.
    dataset = LeRobotDataset(DATASET_ID, download_videos=True)

    print("Initializing RobotKinematics...")
    kinematics = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="gripper_frame_link",
        joint_names=joint_names,
    )
    import yaml
    config_path = Path(__file__).parent.parent / "config" / "robot_config.yaml"
    wrist_offset = 0.0
    home_pose = None
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        wrist_offset = float(cfg.get("robot", {}).get("wrist_roll_offset", 0.0))
        home_pose = cfg.get("robot", {}).get("home_pose")

    so101 = SO101Control(urdf_path=URDF_PATH, wrist_roll_offset=wrist_offset, home_pose=home_pose)

    # Prepare for output
    new_dir = Path.home() / ".cache" / "lerobot" / OUT_DATASET_ID
    if new_dir.exists():
        print(f"Warning: Output directory {new_dir} already exists. Removing it.")
        shutil.rmtree(new_dir)
    new_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(dataset.root) / "data"
    parquet_files = sorted(list(data_dir.rglob("*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"Could not find parquet files in {data_dir}")

    global_eef_states = []
    global_eef_actions = []

    for parquet_path in parquet_files:
        print(f"Reading parquet: {parquet_path}")
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        if "observation.state" not in df.columns:
            print(f"Warning: {parquet_path} does not contain 'observation.state'. Skipping fk computation.")
            continue

        total_frames = len(df)
        eef_states = []
        eef_actions = []
        has_action = "action" in df.columns

        print(f"Computing Forward Kinematics over {total_frames} frames in {parquet_path.name}...")
        
        for i in tqdm(range(total_frames)):
            state = df.iloc[i]["observation.state"] # Array of joints in degrees (for SOARM101)
            
            # Calculate for observation.state
            state_deg = np.rad2deg(so101.motor_to_rad(state))
            T_matrix = kinematics.forward_kinematics(state_deg)
            pos = T_matrix[:3, 3]
            r6d = mat_to_rotate6d(T_matrix[:3, :3])
            gripper = [state[-1]]
            eef_states.append(np.concatenate([pos, r6d, gripper]))

            # Calculate for action if present
            if has_action:
                action = df.iloc[i]["action"]
                action_deg = np.rad2deg(so101.motor_to_rad(action))
                T_matrix_act = kinematics.forward_kinematics(action_deg)
                pos_act = T_matrix_act[:3, 3]
                r6d_act = mat_to_rotate6d(T_matrix_act[:3, :3])
                gripper_act = [action[-1]]
                eef_actions.append(np.concatenate([pos_act, r6d_act, gripper_act]))

        # Prepare explicitly typed PyArrow data to save space (float32 vs float64)
        # and use FixedSizeList for efficiency.
        
        # We'll build the table from a dictionary of arrays with explicit types
        data_to_write = {}
        schema_fields = []

        # 1. Observation State (10D EEF)
        obs_state_arr = np.array(eef_states, dtype=np.float32).flatten()
        obs_state_fixed = pa.FixedSizeListArray.from_arrays(obs_state_arr, 10)
        data_to_write["observation.state"] = obs_state_fixed
        schema_fields.append(pa.field("observation.state", pa.list_(pa.float32(), 10)))

        # 2. Action (10D EEF)
        if has_action:
            action_arr = np.array(eef_actions, dtype=np.float32).flatten()
            action_fixed = pa.FixedSizeListArray.from_arrays(action_arr, 10)
            data_to_write["action"] = action_fixed
            schema_fields.append(pa.field("action", pa.list_(pa.float32(), 10)))

        # 3. Observation Joint Positions (6D)
        joint_pos_arr = np.array(df["observation.state"].tolist(), dtype=np.float32).flatten()
        joint_pos_fixed = pa.FixedSizeListArray.from_arrays(joint_pos_arr, 6)
        data_to_write["observation.joint_positions"] = joint_pos_fixed
        schema_fields.append(pa.field("observation.joint_positions", pa.list_(pa.float32(), 6)))

        # 4. Action Joints (6D)
        if has_action:
            action_joints_arr = np.array(df["action"].tolist(), dtype=np.float32).flatten()
            action_joints_fixed = pa.FixedSizeListArray.from_arrays(action_joints_arr, 6)
            data_to_write["action_joints"] = action_joints_fixed
            schema_fields.append(pa.field("action_joints", pa.list_(pa.float32(), 6)))

        # 5. Scalar Columns (preserve types from original)
        for col in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
            if col in df.columns:
                data_to_write[col] = pa.array(df[col])
                schema_fields.append(pa.field(col, table.schema.field(col).type))

        new_schema = pa.schema(schema_fields)
        table_out = pa.Table.from_pydict(data_to_write, schema=new_schema)

        global_eef_states.extend(eef_states)
        if has_action:
            global_eef_actions.extend(eef_actions)

        rel_path = parquet_path.relative_to(data_dir)
        new_parquet_path = new_dir / "data" / rel_path
        new_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Writing updated parquet to: {new_parquet_path.relative_to(new_dir)}")
        pq.write_table(table_out, new_parquet_path)

    # Handle metadata files which might be under root or meta/
    state_action_names = ["x", "y", "z", "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5", "gripper"]

    meta_dir = new_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Copy the whole meta directory first to ensure we have tasks.parquet and episodes parquets
    source_meta = Path(dataset.root) / "meta"
    if source_meta.exists():
        shutil.copytree(source_meta, meta_dir, dirs_exist_ok=True)
        print("Copied complete meta directory.")

    info_path = Path(dataset.root) / "meta" / "info.json"
    if not info_path.exists(): info_path = Path(dataset.root) / "info.json"
    
    if info_path.exists():
        with open(info_path, "r") as f:
            info_dict = json.load(f)
        
        if "features" in info_dict:
            # Add joint features and modify state and action
            if "observation.state" in info_dict["features"]:
                old_state = info_dict["features"]["observation.state"].copy()
                info_dict["features"]["observation.joint_positions"] = old_state
                info_dict["features"]["observation.state"]["shape"] = [10]
                info_dict["features"]["observation.state"]["names"] = state_action_names
            
            if "action" in info_dict["features"]:
                old_action = info_dict["features"]["action"].copy()
                info_dict["features"]["action_joints"] = old_action
                info_dict["features"]["action"]["shape"] = [10]
                info_dict["features"]["action"]["names"] = state_action_names

            for k in ["observation.robot_state.eef.mat", "observation.robot_state.eef.pos", "observation.robot_state.eef.rot6d"]:
                if k in info_dict["features"]:
                    del info_dict["features"][k]

        out_info = meta_dir / "info.json"
        with open(out_info, "w") as f:
            json.dump(info_dict, f, indent=4)
        print("Updated info.json automatically.")
    
    features_path = Path(dataset.root) / "meta" / "features.json"
    if not features_path.exists(): features_path = Path(dataset.root) / "features.json"
    
    if features_path.exists():
        with open(features_path, "r") as f:
            features_dict = json.load(f)
            
        if "observation.state" in features_dict:
            features_dict["observation.joint_positions"] = features_dict["observation.state"].copy()
            features_dict["observation.state"]["shape"] = [10]
            features_dict["observation.state"]["names"] = state_action_names
        
        if "action" in features_dict:
            features_dict["action_joints"] = features_dict["action"].copy()
            features_dict["action"]["shape"] = [10]
            features_dict["action"]["names"] = state_action_names

        for k in ["observation.robot_state.eef.mat", "observation.robot_state.eef.pos", "observation.robot_state.eef.rot6d"]:
            if k in features_dict:
                del features_dict[k]
            
        out_features = meta_dir / "features.json"
        with open(out_features, "w") as f:
            json.dump(features_dict, f, indent=4)
        print("Updated features.json automatically.")

    # Recompute stats.json
    def compute_stats(data_list):
        arr = np.array(data_list, dtype=np.float32)
        return {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0, ddof=1).tolist(),
            "count": [len(arr)],
            "q01": np.percentile(arr, 1, axis=0).tolist(),
            "q10": np.percentile(arr, 10, axis=0).tolist(),
            "q50": np.percentile(arr, 50, axis=0).tolist(),
            "q90": np.percentile(arr, 90, axis=0).tolist(),
            "q99": np.percentile(arr, 99, axis=0).tolist(),
        }

    stats_path = Path(dataset.root) / "meta" / "stats.json"
    if not stats_path.exists(): stats_path = Path(dataset.root) / "stats.json"

    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats_dict = json.load(f)

        if "observation.state" in stats_dict and len(global_eef_states) > 0:
            stats_dict["observation.joint_positions"] = stats_dict["observation.state"].copy()
            stats_dict["observation.state"] = compute_stats(global_eef_states)
        
        if "action" in stats_dict and len(global_eef_actions) > 0:
            stats_dict["action_joints"] = stats_dict["action"].copy()
            stats_dict["action"] = compute_stats(global_eef_actions)

        out_stats = meta_dir / "stats.json"
        with open(out_stats, "w") as f:
            json.dump(stats_dict, f, indent=4)
        print("Updated stats.json automatically.")
    
    # Copy all other directories from the source root to the new dataset (e.g. videos, images, calibration)
    # This ensures we don't miss any folder that might be interpreted as "images" by the user.
    for item in Path(dataset.root).iterdir():
        if item.is_dir() and item.name not in ["data", "meta", ".cache", ".git"]:
            target_item = new_dir / item.name
            if not target_item.exists():
                shutil.copytree(item, target_item, dirs_exist_ok=True)
                print(f"Copied {item.name} directory to the new dataset folder.")

    readme_path = Path(dataset.root) / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()
        
        # Update references to the old dataset ID to reflect the new 6D dataset URL and ID
        new_readme_content = readme_content.replace(DATASET_ID, OUT_DATASET_ID)
        
        # LeRobot hardcodes the info.json into the README dataset card. We need to overwrite it!
        import re
        if "info_dict" in locals():
            new_json_str = json.dumps(info_dict, indent=4)
            new_readme_content = re.sub(
                r"```json\n.*?\n```", 
                f"```json\n{new_json_str}\n```", 
                new_readme_content, 
                flags=re.DOTALL
            )
        
        with open(new_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(new_readme_content)
            
        print("Copied and updated dataset card (README.md).")

    print(f"New dataset prepared at: {new_dir}")
    print("info.json and features.json have been successfully updated.")

    if PUSH_TO_HUB:
        print(f"Pushing {OUT_DATASET_ID} to the Hugging Face hub...")
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=OUT_DATASET_ID, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=new_dir,
            repo_id=OUT_DATASET_ID,
            repo_type="dataset",
        )
        try:
            version = info_dict.get("codebase_version", "v3.0")
            api.create_tag(repo_id=OUT_DATASET_ID, tag=version, repo_type="dataset")
            print(f"Successfully tagged dataset with version {version}!")
        except Exception as e:
            print(f"Warning: Could not tag dataset version: {e}")
            
        print(f"Successfully pushed {OUT_DATASET_ID} to the hub!")


if __name__ == "__main__":
    main()
