#!/usr/bin/env python3
import cv2
import speech_recognition as sr
import threading
from queue import Queue, Empty
import numpy as np
from collections import deque
import time
import openvino as ov
import os

class RealtimeSegmentation:
    def __init__(self, yolo_model_path, sam_encoder_path, sam_decoder_path):
        print("Starting initialization...")

        # Initialize OpenVINO runtime
        core = ov.Core()
        print("OpenVINO Core initialized")

        # Check for GPU availability and set device
        if "GPU" in core.available_devices:
            self.device = "GPU"
            print("Using Intel GPU")
        else:
            self.device = "CPU"
            print("GPU not available, using CPU")

        print(f"Using device: {self.device}")
        
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        print("Loading YOLO model...")
        self.yolo_model = core.compile_model(yolo_model_path, device_name=self.device)
        self.yolo_input_layer = self.yolo_model.input()
        self.yolo_output_layer = self.yolo_model.output()
        print("YOLO model loaded successfully")

        print("Loading SAM models...")
        self.sam_encoder = core.compile_model(sam_encoder_path, device_name=self.device)
        self.sam_decoder = core.compile_model(sam_decoder_path, device_name='CPU')
        print("SAM models loaded successfully")

        print("Initializing speech recognizer...")
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()

        print("Setting up queues...")
        self.command_queue = Queue()
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue()
        self.last_masks = deque(maxlen=3)  # Reduced from 5 to 3 for better performance
        self.tracking_active = False
        self.target_class = None

        print("Initialization complete!")
    
    def process_sam_image(self, frame):
        """Process image for SAM model"""
        original_h, original_w = frame.shape[:2]

        # Directly resize to 1024x1024
        input_size = 1024
        final_image = cv2.resize(frame, (input_size, input_size))

        # Calculate scale factors for coordinate mapping
        x_scale = input_size / original_w
        y_scale = input_size / original_h

        # Convert to float32 and normalize to 0-1
        input_image = final_image.astype(np.float32) / 255.0
        input_image = input_image.transpose((2, 0, 1))[None, ...]

        return input_image, (x_scale, y_scale)

    def preprocess_yolo_image(self, frame):
        """Preprocess frame for YOLO model"""
        input_height, input_width = 640, 640
        resized = cv2.resize(frame, (input_width, input_height))
        input_image = np.array(resized, dtype=np.float32) / 255.0
        input_image = input_image.transpose((2, 0, 1))
        input_image = np.expand_dims(input_image, 0)
        return input_image

    def process_yolo_output(self, output, frame_shape):
        """Process YOLO output and return detections"""
        output = output[0]
        frame_height, frame_width = frame_shape[:2]
        
        predictions = output  # Already in correct format from our converter
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        mask = confidences > 0.3
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        results = []
        for box, class_id, conf in zip(boxes, class_ids, confidences):
            x, y, w, h = box
            
            # Scale from YOLO input size to frame size
            x = x * frame_width / 640
            y = y * frame_height / 640
            w = w * frame_width / 640
            h = h * frame_height / 640
            
            x1 = int(max(0, x - w/2))
            y1 = int(max(0, y - h/2))
            x2 = int(min(frame_width, x + w/2))
            y2 = int(min(frame_height, y + h/2))
            
            if x2 > x1 and y2 > y1:
                results.append({
                    'bbox': [x1, y1, x2, y2],
                    'class_id': int(class_id),
                    'conf': float(conf)
                })

        return results

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

    def segmentation_worker(self):
        print("Starting segmentation worker...")
        last_process_time = time.time()
        last_print_time = time.time()

        while True:
            current_time = time.time()
            if current_time - last_process_time < 0.2:  # 5 FPS processing
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
                    input_tensor = self.preprocess_yolo_image(frame)
                    results = self.yolo_model([input_tensor])[self.yolo_output_layer]
                    detections = self.process_yolo_output(results, frame.shape)
                    
                    # Filter detections for target class
                    target_detections = [
                        det for det in detections 
                        if self.class_names[det['class_id']] == self.target_class 
                        and det['conf'] > 0.5
                    ]
                    
                    if target_detections:
                        # Get highest confidence detection
                        best_detection = max(target_detections, key=lambda x: x['conf'])
                        if current_time - last_print_time > 1.0:
                            print(f"\rTracking {self.target_class}: {best_detection['conf']:.2f}", end="")
                            last_print_time = current_time
                        
                        # Get box center for SAM
                        bbox = best_detection['bbox']
                        center_x = int((bbox[0] + bbox[2]) / 2)
                        center_y = int((bbox[1] + bbox[3]) / 2)
                        
                        # Process image for SAM - using simpler resize strategy
                        sam_input, scale_factors = self.process_sam_image(frame)
                        image_embeddings = self.sam_encoder([sam_input])[0]
                        
                        # Get center point and scale to SAM's image space
                        x_scale, y_scale = scale_factors
                        center_x = int((bbox[0] + bbox[2]) / 2)
                        center_y = int((bbox[1] + bbox[3]) / 2)
                        point_coords = np.array([[[
                            center_x * x_scale,
                            center_y * y_scale
                        ]]], dtype=np.float32)
                        point_labels = np.array([[1]], dtype=np.float32)
                        
                        # Run SAM decoder
                        decoder_outputs = self.sam_decoder([
                            image_embeddings,
                            point_coords,
                            point_labels
                        ])
                        
                        masks = decoder_outputs[0]
                        iou_predictions = decoder_outputs[1]
                        
                        # Get best mask
                        mask = masks[0, np.argmax(iou_predictions[0])]
                        
                        # Resize mask to original size
                        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0])) > 0.5
                        
                        # Apply temporal smoothing
                        self.last_masks.append(mask)
                        if len(self.last_masks) > 1:
                            smooth_mask = np.mean(self.last_masks, axis=0) > 0.5
                        else:
                            smooth_mask = mask
                        
                        self.result_queue.put({
                            'bbox': bbox,
                            'mask': smooth_mask,
                            'class': self.target_class,
                            'conf': float(best_detection['conf'])
                        })
                
                last_process_time = current_time
                    
            except Empty:
                continue
            except Exception as e:
                if str(e):
                    print(f"\nSegmentation error: {e}")
                import traceback
                traceback.print_exc()

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
                        mask_overlay = display_frame.copy()
                        mask_overlay[last_result['mask']] = [0, 255, 0]
                        cv2.addWeighted(mask_overlay, 0.4, display_frame, 0.6, 0, display_frame)

                bbox = last_result['bbox']
                if all(x >= 0 for x in bbox):
                    cv2.rectangle(display_frame, 
                                (bbox[0], bbox[1]), 
                                (bbox[2], bbox[3]), 
                                (0, 255, 0), 2)
                    
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
    print(os.path.exists("openvino_models/mobile_sam_encoder.xml"))
    print(os.path.exists("openvino_models/mobile_sam_decoder.xml"))
    print(os.path.exists("openvino_models/yolov8n.xml"))
    sam_encoder_path = "openvino_models/mobile_sam_encoder.xml"
    sam_decoder_path = "openvino_models/mobile_sam_decoder.xml"
    yolo_model_path = "openvino_models/yolov8n.xml"
    
    try:
        detector = RealtimeSegmentation(yolo_model_path, sam_encoder_path, sam_decoder_path)
        detector.run()
    except Exception as e:
        print(f"\nError during initialization: {e}")
        print("\nFull error trace:")
        import traceback
        traceback.print_exc()