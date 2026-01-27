import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

print("Loading SmolVLA baseline model...")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

print("\n" + "="*60)
print("NORMALIZATION STATISTICS INSPECTION")
print("="*60)

# Check normalize_inputs module
if hasattr(policy, 'normalize_inputs'):
    print("\n[INPUT NORMALIZATION MODULE]")
    norm_module = policy.normalize_inputs
    print(f"Normalization module type: {type(norm_module).__name__}")
    
    if hasattr(norm_module, 'normalization_modes'):
        print(f"Normalization modes: {norm_module.normalization_modes}")
    
    if hasattr(norm_module, 'stats'):
        for key, stats in norm_module.stats.items():
            print(f"\n{key}:")
            if hasattr(stats, 'mean'):
                mean_val = stats.mean.tolist() if isinstance(stats.mean, torch.Tensor) else stats.mean
                print(f"  mean: {mean_val}")
                if isinstance(stats.mean, torch.Tensor) and torch.isinf(stats.mean).any():
                    print(f"  **[WARNING]** mean contains inf values!")
            if hasattr(stats, 'std'):
                std_val = stats.std.tolist() if isinstance(stats.std, torch.Tensor) else stats.std
                print(f"  std: {std_val}")
                if isinstance(stats.std, torch.Tensor) and torch.isinf(stats.std).any():
                    print(f"  **[WARNING]** std contains inf values!")

# Check normalize_targets module (for actions)
if hasattr(policy, 'normalize_targets'):
    print("\n[ACTION/OUTPUT NORMALIZATION MODULE]")
    norm_module = policy.normalize_targets
    print(f"Normalization module type: {type(norm_module).__name__}")
    
    if hasattr(norm_module, 'normalization_modes'):
        print(f"Normalization modes: {norm_module.normalization_modes}")
    
    if hasattr(norm_module, 'stats'):
        for key, stats in norm_module.stats.items():
            print(f"\n{key}:")
            if hasattr(stats, 'mean'):
                mean_val = stats.mean.tolist() if isinstance(stats.mean, torch.Tensor) else stats.mean
                print(f"  mean: {mean_val}")
                if isinstance(stats.mean, torch.Tensor) and torch.isinf(stats.mean).any():
                    print(f"  **[WARNING]** mean contains inf values!")
            if hasattr(stats, 'std'):
                std_val = stats.std.tolist() if isinstance(stats.std, torch.Tensor) else stats.std
                print(f"  std: {std_val}")
                if isinstance(stats.std, torch.Tensor) and torch.isinf(stats.std).any():
                    print(f"  **[WARNING]** std contains inf values!")

print("\n" + "="*60)
print("INSPECTION COMPLETE")
print("="*60)
