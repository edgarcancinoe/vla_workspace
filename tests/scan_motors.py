#!/usr/bin/env python3
"""Scan for connected motors to diagnose connection issues."""

import yaml
from pathlib import Path
from lerobot.motors.feetech.feetech import FeetechMotorsBus

# Load config
config_path = Path(__file__).parent.parent / "config" / "robot_config.yaml"
with open(config_path, 'r') as f:
    config_data = yaml.safe_load(f)

port = config_data["robot"]["port"]
print(f"Scanning motors on port: {port}")
print("="*60)

# Create motor bus
bus = FeetechMotorsBus(port=port)

print("\nScanning for motors (IDs 1-10)...")
print("-"*60)

found_motors = []
for motor_id in range(1, 11):
    try:
        # Try to read Position from each ID
        result = bus.read("Present_Position", motor_id, num_retry=1)
        if result is not None:
            print(f"✓ Motor ID {motor_id}: FOUND (Position: {result})")
            found_motors.append(motor_id)
    except Exception as e:
        print(f"✗ Motor ID {motor_id}: NOT FOUND")

print("-"*60)
print(f"\nSummary: Found {len(found_motors)} motors: {found_motors}")
print("\nExpected motors for SO-100:")
print("  ID 1: shoulder_pan")
print("  ID 2: shoulder_lift")
print("  ID 3: elbow_flex")
print("  ID 4: wrist_flex")
print("  ID 5: wrist_roll")
print("  ID 6: gripper")
print("="*60)

bus.disconnect()
