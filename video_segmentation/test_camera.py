#!/usr/bin/env python3
import cv2
import time

def test_camera_access():
    # List of backends to try
    backends = [
        (cv2.CAP_ANY, "Default"),
        (cv2.CAP_DSHOW, "DirectShow")
    ]
    
    for backend_id, backend_name in backends:
        print(f"\nTesting {backend_name} backend:")
        
        for i in range(5):  # Try first 5 indices
            try:
                print(f"Attempting to open camera {i}...")
                cap = cv2.VideoCapture(i + backend_id)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        print(f"Successfully opened and read from camera {i}")
                        print(f"Frame shape: {frame.shape if frame is not None else 'None'}")
                        
                        # Try to display the frame
                        cv2.imshow('Test Frame', frame)
                        cv2.waitKey(1000)  # Show for 1 second
                        cv2.destroyAllWindows()
                    else:
                        print(f"Opened camera {i} but couldn't read frame")
                else:
                    print(f"Couldn't open camera {i}")
                    
            except Exception as e:
                print(f"Error testing camera {i}: {str(e)}")
            
            finally:
                if 'cap' in locals():
                    cap.release()
            
            time.sleep(0.5)  # Short delay between attempts

if __name__ == "__main__":
    print("Starting camera test...")
    print("OpenCV version:", cv2.__version__)
    test_camera_access()
    print("\nTest complete")