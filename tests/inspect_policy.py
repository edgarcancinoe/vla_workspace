from lerobot.policies import SmolVLAConfig
from lerobot.policies.factory import get_policy_class
import sys
import torch

def print_feature_details(features, title):
    print(f"\n{title}:")
    if not features:
        print("  None")
        return

    for key, feature in features.items():
        shape = feature.shape
        dtype = getattr(feature, "dtype", "Unknown")
        print(f"  - {key}:")
        print(f"      Shape: {shape}")
        print(f"      Type:  {dtype}")

def print_normalization_stats(policy):
    print("\nNORMALIZATION STATS:")
    if hasattr(policy, "normalization_stats") and policy.normalization_stats:
        for key, stats in policy.normalization_stats.items():
            print(f"  - {key}:")
            for stat_name, value in stats.items():
                if isinstance(value, torch.Tensor):
                    # concise print for tensors
                    val_str = f"Tensor shape={value.shape}"
                    if value.numel() < 5:
                         val_str += f" values={value.tolist()}"
                    print(f"      {stat_name}: {val_str}")
                else:
                    print(f"      {stat_name}: {value}")
    else:
        print("  No normalization stats found.")

try:
    # Try to load the config from the pretrained path
    pretrained_path = "lerobot/smolvla_base"
    print(f"Loading config from {pretrained_path}...")
    
    # We can use the factory or direct class
    policy_class = get_policy_class("smolvla")
    policy = policy_class.from_pretrained(pretrained_path)
    
    print("-" * 50)
    print(f"Policy Type: {type(policy).__name__}")
    
    # Inputs
    print_feature_details(policy.config.input_features, "EXPECTED INPUT FEATURES")
    
    # Outputs
    print_feature_details(policy.config.output_features, "EXPECTED OUTPUT FEATURES")

    # Normalization
    print_normalization_stats(policy)
    
    # Text / Language Capabilities
    print("-" * 50)
    print("LANGUAGE CAPABILITIES:")
    if hasattr(policy, "tokenizer") and policy.tokenizer:
        print(f"  Tokenizer found: {type(policy.tokenizer).__name__}")
        print(f"  Vocab size: {policy.tokenizer.vocab_size if hasattr(policy.tokenizer, 'vocab_size') else 'Unknown'}")
    else:
        print("  No tokenizer attribute found on policy.")
        
    # Check config for text settings
    text_attrs = [k for k in dir(policy.config) if "text" in k.lower() or "vocab" in k.lower() or "max_length" in k.lower()]
    if text_attrs:
        print("  Text-related config:")
        for attr in text_attrs:
            val = getattr(policy.config, attr)
            if not str(attr).startswith("_") and not callable(val):
                 print(f"    - {attr}: {val}")
    
    print("-" * 50)

except Exception as e:
    print(f"Error loading policy: {e}")
    import traceback
    traceback.print_exc()
