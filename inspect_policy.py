from lerobot.policies import SmolVLAConfig
from lerobot.policies.factory import get_policy_class
import sys

try:
    # Try to load the config from the pretrained path
    pretrained_path = "lerobot/smolvla_base"
    print(f"Loading config from {pretrained_path}...")
    
    # We can use the factory or direct class
    policy_class = get_policy_class("smolvla")
    policy = policy_class.from_pretrained(pretrained_path)
    
    print("\nEXPECTED IMAGE FEATURES:")
    for key, feature in policy.config.input_features.items():
        if "image" in key:
            print(f"  - {key}: {feature.shape}")

except Exception as e:
    print(f"Error loading policy: {e}")
