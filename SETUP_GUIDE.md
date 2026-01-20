# VLA Workspace - Client/Server Setup Guide

## Architecture
- **Policy Server**: Runs on `flexo` cluster (GPU machine) - computes actions
- **Robot Client**: Runs on your Mac (connected to physical robot) - executes actions
- **SSH Tunnel**: Connects them securely through the proxy

## Setup Instructions

### 1. Start the Policy Server (on flexo cluster)

```bash
# SSH into the cluster
ssh vicos-flexo

# Navigate to workspace
cd ~/vla_workspace

# Pull latest changes
git pull

# Start the server
./run_server.sh
```

The server will start and show:
```
INFO 2026-01-20 15:10:13 y_server.py:430 PolicyServer started on 127.0.0.1:8080
```

### 2. Set Up SSH Tunnel (on your Mac)

**In a new terminal window**, run:
```bash
cd ~/Documents/Academic/EMAI\ Thesis/vla_workspace
./setup_tunnel.sh
```

This establishes a secure tunnel that forwards your local port 8080 to flexo's port 8080.
**Keep this terminal running** - closing it will close the tunnel.

### 3. Start the Robot Client (on your Mac)

**In another terminal window**, run:
```bash
cd ~/Documents/Academic/EMAI\ Thesis/vla_workspace
./run_client.sh
```

The client will:
1. Connect to cameras
2. Connect to robot motors
3. Connect to the policy server (through the tunnel)
4. Start executing actions


### 4. Control the Robot

To tell the robot what to do, edit `launch_client.yaml`:

```yaml
task: "pick up the red block"
```

Change this string to whatever instructions you want to give the VLA model.
*Note: The effectiveness depends on the model's training data and your workspace setup.*

## Troubleshooting


### "Connection refused" error
- ✅ Check that the policy server is running on flexo
- ✅ Check that the SSH tunnel is active (setup_tunnel.sh)
- ✅ Verify tunnel is working: `curl http://localhost:8080` (should get a response)

### "No motors found" error
- ✅ Check that robot motors are powered on
- ✅ Verify USB connections to U2D2 adapters
- ✅ Run `python test_motors.py` to scan for motors

### Camera connection issues
- ✅ Check camera indices (0 for laptop, 1 for external)
- ✅ Close other apps using the cameras (Zoom, Skype, etc.)

## File Structure

- `launch_client.yaml` - Robot client configuration
- `launch_server.yaml` - Policy server configuration  
- `run_client.sh` - Start robot client on Mac
- `run_server.sh` - Start policy server on flexo
- `setup_tunnel.sh` - Establish SSH tunnel
- `test_motors.py` - Diagnostic script for motor detection

## Network Flow

```
[Mac: Robot Client]
      ↓
[Mac: localhost:8080] 
      ↓ (SSH tunnel)
[proxy.vicos.si:30659]
      ↓
[flexo: localhost:8080]
      ↓
[flexo: Policy Server]
```
