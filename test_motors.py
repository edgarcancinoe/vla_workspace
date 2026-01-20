#!/usr/bin/env python3
"""Scan for Dynamixel motors on connected USB ports."""
import dynamixel_sdk as dxl

# Protocol version
PROTOCOL_VERSION = 2.0
BAUDRATE = 1_000_000  # Common baudrate for Dynamixel

def scan_port(port_name):
    """Scan a port for Dynamixel motors."""
    print(f"\n=== Scanning {port_name} ===")
    
    # Initialize PortHandler and PacketHandler
    port_handler = dxl.PortHandler(port_name)
    packet_handler = dxl.PacketHandler(PROTOCOL_VERSION)
    
    # Open port
    if not port_handler.openPort():
        print(f"Failed to open port {port_name}")
        return []
    
    # Set baudrate
    if not port_handler.setBaudRate(BAUDRATE):
        print(f"Failed to set baudrate to {BAUDRATE}")
        port_handler.closePort()
        return []
    
    print(f"Port opened successfully at {BAUDRATE} baud")
    
    # Scan for motors (IDs 0-253)
    found_motors = []
    print("Scanning motor IDs 1-10 (quick scan)...")
    
    for motor_id in range(1, 11):  # Quick scan first 10 IDs
        # Try to ping the motor
        model_number, result, error = packet_handler.ping(port_handler, motor_id)
        
        if result == dxl.COMM_SUCCESS:
            print(f"Found motor ID {motor_id}, Model: {model_number}")
            found_motors.append((motor_id, model_number))
    
    port_handler.closePort()
    
    if not found_motors:
        print("No motors found in IDs 1-10")
    
    return found_motors

if __name__ == "__main__":
    ports = [
        "/dev/tty.usbmodem5AB90655421",  # Robot (follower)
        "/dev/tty.usbmodem5AAF2198491",  # Teleop (leader)
    ]
    
    for port in ports:
        motors = scan_port(port)
        if motors:
            print(f"\n{port}: Found {len(motors)} motor(s)")
            for motor_id, model in motors:
                print(f"  - ID {motor_id}: Model {model}")

