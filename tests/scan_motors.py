
import argparse
from lerobot.motors.feetech.feetech import FeetechMotorsBus

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, required=True, help="Port to scan")
    args = parser.parse_args()

    print(f"Scanning port {args.port}...")
    
    # Try initializing bus with no motors to just use scan capabilities
    # We might need to mock or provide minimal args
    try:
        bus = FeetechMotorsBus(port=args.port, motors={})
        if not bus.port_handler.openPort():
            print("Failed to open port")
            return
        if not bus.port_handler.setBaudRate(1000000):
            print("Failed to set baudrate")
            return
            
        ids = bus.broadcast_ping()
        print("Found motor IDs:", ids)
        
        if IDs := ids:
             print("\nDetailed scan:")
             for id_ in IDs:
                 print(f"ID: {id_}, Model: {IDs[id_]}")
        else:
             print("No motors found via broadcast ping.")

    except Exception as e:
        print(f"Error scanning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
