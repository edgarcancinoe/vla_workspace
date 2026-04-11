import torch
import sys
import logging
from termcolor import colored

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

def debug_batch(batch, tag="", step=0, only_step=0, slice_dim=None, dataset_meta=None):
    if step != only_step:
        return
    
    logging.info(colored(f"\n{'='*70}", "magenta", attrs=["bold"]))
    logging.info(colored(f"[DEBUG] {tag} | step={step}", "magenta", attrs=["bold"]))
    logging.info(colored(f"{'='*70}", "magenta", attrs=["bold"]))
    
    if not batch:
        logging.info(colored("  (Empty dictionary)", "red"))
        logging.info("")
        return

    ignore_keys = {"index", "info", "episode_index"}

    def sort_key(k):
        if k == "action": return (0, k)
        if k == "observation.state": return (1, k)
        if k == "action_is_pad": return (2, k)
        return (3, k)

    sorted_keys = sorted([k for k in batch.keys() if k not in ignore_keys], key=sort_key)

    for k in sorted_keys:
        v = batch[k]
        padded_key = f"  {k}:".ljust(40)
        key_str = colored(padded_key, "cyan")
        
        if isinstance(v, torch.Tensor):
            shape_str = f"shape={list(v.shape)} dtype={v.dtype}"
            
            if k in ["action", "observation.state", "pred_action"] and v.numel() > 0 and v.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                logging.info(f"{key_str} {shape_str}")
                
                num_dims = v.size(-1)
                
                feat_names = None
                if dataset_meta and k in dataset_meta.features:
                    feature_info = dataset_meta.features[k]
                    if isinstance(feature_info, dict):
                        feat_names = feature_info.get("names")
                    elif hasattr(feature_info, "names"):
                        feat_names = feature_info.names
                
                name_width = 15
                header_str = f"    {'Dim Name':<{name_width}} | {'min':<8} | {'max':<8} | {'mean':<8}"
                logging.info(colored(f"    {'-' * (len(header_str) - 4)}", "green"))
                logging.info(colored(header_str, "green", attrs=["bold"]))
                logging.info(colored(f"    {'-' * (len(header_str) - 4)}", "green"))
                
                v_flat = v.reshape(-1, num_dims).float()
                v_min = v_flat.min(dim=0).values
                v_max = v_flat.max(dim=0).values
                v_mean = v_flat.mean(dim=0)
                
                for i in range(num_dims):
                    dim_name = f"Dim {i}"
                    if feat_names is not None and i < len(feat_names):
                        d_name = feat_names[i]
                        if isinstance(d_name, dict) and 'name' in d_name:
                            dim_name = d_name['name']
                        elif isinstance(d_name, str):
                            dim_name = d_name
                    elif num_dims == 10 and k in ["action", "observation.state", "pred_action"]:
                        default_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper", "pad_1", "pad_2", "pad_3"]
                        dim_name = default_names[i]
                    elif num_dims == 16 and k in ["action", "observation.state", "pred_action"]:
                        default_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper", "mobile_v", "mobile_w", "pad_1", "pad_2", "pad_3", "pad_4", "pad_5", "pad_6", "pad_7"]
                        if i < len(default_names): dim_name = default_names[i]
                    elif num_dims == 20 and k in ["action", "observation.state", "pred_action"]:
                        default_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper", "pad_1", "pad_2", "pad_3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13"]
                        if i < len(default_names): dim_name = default_names[i]
                    
                    if len(dim_name) > name_width:
                        dim_name = dim_name[:name_width-2] + ".."
                        
                    row_str = f"    {dim_name:<{name_width}} | {v_min[i].item():>8.4f} | {v_max[i].item():>8.4f} | {v_mean[i].item():>8.4f}"
                    logging.info(colored(row_str, "green"))
                logging.info(colored(f"    {'-' * (len(header_str) - 4)}", "green"))
                
            else:
                extra = ""
                if v.numel() > 0 and v.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                    extra = f" | min={v.min().item():.4f} max={v.max().item():.4f} mean={v.mean().item():.4f}"
                elif v.numel() == 0:
                    extra = " | (empty tensor)"
                
                logging.info(f"{key_str} {shape_str}{extra}")
        else:
            val_str = str(v)
            if len(val_str) > 60:
                val_str = val_str[:57] + "..."
            logging.info(f"{key_str} type={type(v).__name__} value={val_str}")
            
    logging.info("")


batch = {
    "action": torch.randn(8, 30, 16),
    "action_is_pad": torch.ones(8, 30, dtype=torch.bool),
    "episode_index": torch.tensor([5, 5, 5, 5, 5, 5, 5, 5]),
    "index": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
    "observation.images.image": torch.randn(8, 3, 480, 640),
    "observation.images.image2": torch.randn(8, 3, 480, 640),
    "observation.state": torch.randn(8, 16),
    "domain_id": torch.zeros(8, dtype=torch.int64)
}

class FakeFeatures:
    names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper", "mobile_v", "mobile_w", "pad_1", "pad_2", "pad_3", "pad_4", "pad_5", "pad_6", "pad_7"]

class FakeMeta:
    features = {
        "action": FakeFeatures(),
        "observation.state": FakeFeatures()
    }

debug_batch(batch, tag="POST (after preprocess)", step=0, dataset_meta=FakeMeta())
