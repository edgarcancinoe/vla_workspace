import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

print("Testing imports...")
from lerobot.processor import make_default_processors
print("lerobot.processor imported")
from robot_control.so101_control import SO101Control
print("SO101Control imported")
print("All imports done.")
