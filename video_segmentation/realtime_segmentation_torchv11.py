#!/usr/bin/env python3
import cv2
import speech_recognition as sr
import threading
from queue import Queue, Empty
import numpy as np
from collections import deque, defaultdict
import time
from ultralytics import YOLO

class RealtimeSegmentation:
    def __init__(self):
        print("Starting initialization...")
        
        # Initialize YOLO11-seg model
        print("Loading YOLO11-seg model...")
        self.model = YOLO('yolo11n-seg.pt')
        print("Model loaded successfully!")
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        
        # Queue setup
        self.command_queue = Queue()
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue()
        
        # Track history for each object ID
        self.track_history = defaultdict(lambda: deque(maxlen=2))
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
            if current_time - last_process_time < 0.0165:  # 60 FPS processing
                continue
                
            try:
                frame = self.frame_queue.get_nowait()
                command = None
                
                # Process voice commands
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
                    # Run YOLO11-seg with tracking
                    results = self.model.track(frame, persist=True, verbose=False)
                    
                    for r in results:
                        if r.boxes.id is not None:  # If we have tracking IDs
                            track_ids = r.boxes.id.cpu().numpy().astype(int)
                            for idx, track_id in enumerate(track_ids):
                                cls = r.names[int(r.boxes.cls[idx])]
                                
                                if cls == self.target_class:
                                    conf = r.boxes.conf[idx].item()
                                    if conf > 0.5:
                                        if current_time - last_print_time > 1.0:
                                            print(f"\rTracking {cls} ID {track_id}: {conf:.2f}", end="")
                                            last_print_time = current_time
                                        
                                        # Get mask and box
                                        mask = r.masks.data[idx].cpu().numpy()
                                        box = r.boxes.xyxy[idx].cpu().numpy()
                                        
                                        # Apply temporal smoothing per tracked object
                                        self.track_history[track_id].append(mask)
                                        smooth_mask = np.mean(list(self.track_history[track_id]), axis=0) > 0.5
                                        
                                        self.result_queue.put({
                                            'bbox': box.tolist(),
                                            'mask': smooth_mask,
                                            'class': cls,
                                            'conf': conf,
                                            'track_id': track_id
                                        })
                
                last_process_time = current_time
                    
            except Empty:
                continue
            except Exception as e:
                print(f"\nSegmentation error: {e}")
                import traceback
                traceback.print_exc()

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
            
            try:
                result = self.result_queue.get_nowait()
                if result:
                    last_result = result
            except Empty:
                pass
            except Exception as e:
                print(f"\nVisualization error: {e}")
            
            if last_result:
                if last_result['mask'] is not None:
                    if last_result['mask'].shape[:2] == frame.shape[:2]:
                        overlay = display_frame.copy()
                        overlay[last_result['mask']] = overlay[last_result['mask']] * [0.3, 1, 0.3]
                        cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)
                
                bbox = last_result['bbox']
                if all(x >= 0 for x in bbox):
                    cv2.rectangle(display_frame, 
                                (int(bbox[0]), int(bbox[1])), 
                                (int(bbox[2]), int(bbox[3])), 
                                (0, 255, 0), 2)
                    
                    label = f"{last_result['class']} {last_result['conf']:.2f} ID:{last_result['track_id']}"
                    cv2.putText(display_frame, 
                              label, 
                              (int(bbox[0]), int(bbox[1]) - 10), 
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
    try:
        detector = RealtimeSegmentation()
        detector.run()
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting steps:")
        print("1. Check if your camera is connected")
        print("2. Check if you have yolo11n-seg.pt downloaded")
        print("3. Check if you have the latest ultralytics installed")