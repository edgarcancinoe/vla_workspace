# VLA Robot Control

This workspace uses `lerobot.async_inference` to control an SO-101 robot via a remote GPU.

## Quick Start

### 1. The Tunnel (Local Mac)
In a fresh terminal on Mac:
```bash
ssh -L 8080:localhost:8080 -p 30660 jose@proxy.vicos.si
```

### 2. The Server (Remote GPU)
In the remote shell (after SSH):
```bash
conda activate thesis
python -m lerobot.async_inference.policy_server --config_path launch_server.yaml
```

### 3. The Client (Local Mac)
In another local terminal:
```bash
conda activate thesis
python -m lerobot.async_inference.robot_client --config_path launch_client.yaml
```

## Configuration
Edit `launch_client.yaml` to change:
- `pretrained_name_or_path`: Model to use (e.g. `lerobot/smolvla_base`)
- `robot.port`: The USB port of your robot
- `server_address`: Change if not using a tunnel (e.g. `10.x.x.x:8080`)
