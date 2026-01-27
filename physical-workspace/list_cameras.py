import cv2

def list_cameras():
    index = 0
    arr = []
    while index < 10:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            print(f"Camera index {index} is available.")
            arr.append(index)
            cap.release()
        else:
            print(f"Camera index {index} is not available.")
        index += 1
    return arr

if __name__ == "__main__":
    print("Listing available cameras...")
    available = list_cameras()
    print(f"Available camera indices: {available}")
