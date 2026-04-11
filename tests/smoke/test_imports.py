import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

print("Testing imports...")
from lerobot.processor import make_default_processors
print("lerobot.processor imported")
from thesis_vla.robot.so101_control import SO101Control
print("SO101Control imported")
print("All imports done.")
