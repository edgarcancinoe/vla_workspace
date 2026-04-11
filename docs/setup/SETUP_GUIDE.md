# Setup Guide

## Robot client/server workflow

Run commands from the workspace root.

### Policy server

Use:

```bash
./apps/robot/run_server.sh
```

### SSH tunnel

Use:

```bash
./apps/robot/setup_tunnel.sh
```

### Robot client

Use:

```bash
./apps/robot/run_client.sh
```

## Canonical config locations

- robot config: `config/robot/robot_config.yaml`
- client launch config: `config/launch/launch_client.yaml`
- server launch config: `config/launch/launch_server.yaml`

## Training launchers

- XVLA: `python apps/train/launch_finetune_xvla.py`
- SmolVLA: `python apps/train/launch_finetune_smolvla.py`
