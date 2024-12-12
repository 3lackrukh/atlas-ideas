#!/usr/bin/env python3
import cv2
import torch
import speech_recognition as sr
import threading
from queue import Queue, Empty
from mobile_sam import sam_model_registry, SamPredictor
from ultralytics import YOLO
import numpy as np
from collections import deque
import time

class RealtimeSegmentation:
    def __init__(self, sam_checkpoint):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize models
        print("Loading models...")
        self.yolo = YOLO('yolov8n.pt')
        self.sam = sam_model_registry["vit_t"](checkpoint="mobile_sam.pt")
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        print("Models loaded successfully!")
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        
        self.command_queue = Queue()
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue()
        self.last_masks = deque(maxlen=3)  # Reduced from 5 to 3 for better performance
        self.tracking_active = False
        self.target_class = None

    def speech_worker(self):
        """Listen for speech commands continuously"""
        print("Starting speech recognition...")
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source)
            
            while True:
                try:
                    audio = self.recognizer.listen(source, timeout=1)
                    text = self.recognizer.recognize_google(audio)
                    print(f"Heard command: {text}")
                    self.command_queue.put(text.lower())
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    print(f"Speech recognition error: {e}")

    def find_camera(self):
        """Try different camera backends and indices"""
        # Try DirectShow (Windows) backend first
        for i in range(10):
            print(f"Trying camera index {i} with DirectShow...")
            cap = cv2.VideoCapture(i + cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"Successfully opened camera with DirectShow at index {i}")
                    return cap
                cap.release()
    
        # Try default backend as fallback
        for i in range(10):
            print(f"Trying camera index {i} with default backend...")
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"Successfully opened camera at index {i}")
                    return cap
                cap.release()
        return None

    def segmentation_worker(self):
        print("Starting segmentation worker...")
        last_process_time = time.time()
        last_print_time = time.time()
        
        while True:
            current_time = time.time()
            if current_time - last_process_time < 0.2:
                continue
                
            try:
                frame = self.frame_queue.get_nowait()
                command = None
                
                while not self.command_queue.empty():
                    command = self.command_queue.get_nowait()
                    
                    if isinstance(command, str):
                        if "track" in command:
                            self.target_class = command.replace("track", "").strip()
                            self.tracking_active = True
                            print(f"\nStarting to track: {self.target_class}")
                        elif "stop" in command:
                            self.tracking_active = False
                            self.target_class = None
                            print("\nStopping tracking")
                
                if self.tracking_active and self.target_class:
                    # Run YOLO detection
                    results = self.yolo(frame, verbose=False)
                    
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            cls = result.names[box.cls[0].item()]
                            
                            if cls == self.target_class:
                                conf = box.conf[0].item()
                                if conf > 0.5:
                                    if current_time - last_print_time > 1.0:
                                        print(f"\rTracking {cls}: {conf:.2f}", end="")
                                        last_print_time = current_time
                                    
                                    # Get box coordinates
                                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                                    
                                    # Before SAM processing, resize the frame
                                    scale_factor = 0.5
                                    small_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
                                    
                                    # Get center point for SAM
                                    center_x = int((x1 + x2) / 2)
                                    center_y = int((y1 + y2) / 2)
                                    
                                    # Get SAM mask
                                    self.predictor.set_image(small_frame)
                                    input_point = np.array([[center_x * scale_factor, center_y * scale_factor]])
                                    input_label = np.array([1])
                                    
                                    masks, scores, _ = self.predictor.predict(
                                        point_coords=input_point,
                                        point_labels=input_label,
                                        multimask_output=True
                                    )
                                    
                                    # Take highest scoring mask
                                    mask = masks[scores.argmax()]
                                    if mask is not None:
                                        mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0])) > 0
                                        self.last_masks.append(mask)
                                        smooth_mask = np.mean(self.last_masks, axis=0) > 0.5
                                    
                                        self.result_queue.put({
                                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                            'mask': smooth_mask,
                                            'class': cls,
                                            'conf': conf
                                        })
                
                last_process_time = current_time
                    
            except Exception as e:
                if str(e):
                    print(f"\nSegmentation error: {e}")
    
    def run(self):
        threading.Thread(target=self.speech_worker, daemon=True).start()
        threading.Thread(target=self.segmentation_worker, daemon=True).start()
        
        print("Searching for camera...")
        cap = self.find_camera()
        if cap is None:
            raise RuntimeError("No working camera found. Please check your camera connection and permissions.")
        
        cv2.namedWindow('Detection and Segmentation')
        
        print("\nVoice Commands:")
        print("- Say 'track [object]' (e.g., 'track person', 'track cat')")
        print("- Say 'stop' to stop tracking")
        print("- Press 'q' to quit")
        
        last_result = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame, retrying...")
                time.sleep(0.1)
                continue
            
            try:
                self.frame_queue.put_nowait(frame)
            except:
                pass
            
            display_frame = frame.copy()
            
            # Try to get new result
            try:
                result = self.result_queue.get_nowait()
                if result:
                    last_result = result  # Update last result when we get a new one
            except Empty:
                pass
            except Exception as e:
                print(f"\nVisualization error: {e}")
            
            # Always try to draw the last valid result
            if last_result:
                # Draw the segmentation mask
                if last_result['mask'] is not None:
                    if last_result['mask'].shape[:2] == frame.shape[:2]:
                        overlay = display_frame.copy()
                        overlay[last_result['mask']] = overlay[last_result['mask']] * [0.3, 1, 0.3]
                        cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)
                    else:
                        print(f"\nMask shape mismatch: {last_result['mask'].shape} vs {frame.shape}")
                
                # Draw the bounding box
                bbox = last_result['bbox']
                if all(x >= 0 for x in bbox):
                    cv2.rectangle(display_frame, 
                                (bbox[0], bbox[1]), 
                                (bbox[2], bbox[3]), 
                                (0, 255, 0), 2)
                    
                    # Add label with confidence
                    label = f"{last_result['class']} {last_result['conf']:.2f}"
                    cv2.putText(display_frame, 
                              label, 
                              (bbox[0], bbox[1] - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.9, 
                              (0, 255, 0), 
                              2)
            
            cv2.imshow('Detection and Segmentation', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('c'):
                last_result = None
                print("\nCleared tracking display")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    sam_checkpoint = "mobile_sam.pt"
    try:
        detector = RealtimeSegmentation(sam_checkpoint)
        detector.run()
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting steps:")
        print("1. Check if your camera is connected")
        print("2. Run 'ls -l /dev/video*' to list available cameras")
        print("3. Ensure you have proper permissions: 'sudo usermod -a -G video $USER'")
        print("4. Try rebooting if needed")