#!/usr/bin/env python3
import cv2
import speech_recognition as sr
import threading
from queue import Queue, Empty
import numpy as np
from collections import deque, defaultdict
import time
import openvino as ov
from ultralytics.trackers import BOTSORT, BYTETracker
from ultralytics.utils.ops import xyxy2xywh

class DetectionResults:
        def __init__(self, boxes, scores, cls):
            self.boxes = boxes
            self.conf = scores
            self.cls = cls
            self.xywh = boxes  # BYTETracker will use this if xywhr is not present,

class RealtimeSegmentation:
    def __init__(self, model_path):
        print("Starting initialization...")
        
        # Initialize OpenVINO
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
        
        # Load the converted YOLO11-seg model
        print("Loading OpenVINO model...")
        self.model = core.compile_model(model_path, device_name=self.device)
        
        # Get input and output nodes
        self.input_layer = self.model.input(0)
        self.output_det = self.model.output("detection_output")
        self.output_coeff = self.model.output("mask_coefficients_output")
        self.output_proto = self.model.output("mask_prototypes_output")
        print("Model loaded successfully!")
        
        # Initialize YOLO tracker (BoT-SORT by default)
        from types import SimpleNamespace

        tracker_args = SimpleNamespace()
        # BYTETracker parameters (parent class)
        tracker_args.track_buffer = 30
        tracker_args.track_high_thresh = 0.5
        tracker_args.track_low_thresh = 0.1
        tracker_args.new_track_thresh = 0.6
        tracker_args.match_thresh = 0.8
        tracker_args.fuse_score = True
        
        # BOTSORT specific parameters
        tracker_args.proximity_thresh = 0.5
        tracker_args.appearance_thresh = 0.25
        tracker_args.with_reid = False
        tracker_args.gmc_method = 'sparseOptFlow'
        
        # Initialize tracker with complete args
        self.tracker = BOTSORT(args=tracker_args)
        
        # Initialize COCO class names
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
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        
        # Setup queues and tracking state
        self.command_queue = Queue()
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue()
        self.track_history = defaultdict(lambda: deque(maxlen=3))
        self.tracking_active = False
        self.target_class = None

    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        input_height, input_width = 640, 640
        resized = cv2.resize(frame, (input_width, input_height))
        input_image = np.array(resized, dtype=np.float32) / 255.0
        input_image = input_image.transpose((2, 0, 1))
        input_image = np.expand_dims(input_image, 0)
        return input_image, (input_height, input_width)

    def process_outputs(self, det_output, mask_coeffs, mask_protos, orig_shape, input_shape):
        """Process model outputs into detections and masks"""
        # Process detection outputs
        predictions = det_output[0]  # Shape: [8400, 116]
        boxes = predictions[:, :4]  # [8400, 4]
        scores = predictions[:, 4:]  # [8400, 112]

        # Get predictions and filter by confidence first
        confidence_threshold = 0.3
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        mask = confidences > confidence_threshold

        # Filter everything by confidence
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        # Scale boxes to original frame size
        scale_x = orig_shape[1] / input_shape[1]
        scale_y = orig_shape[0] / input_shape[0]

        scaled_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 *= scale_x
            x2 *= scale_x
            y1 *= scale_y
            y2 *= scale_y
            scaled_boxes.append([x1, y1, x2, y2])
        scaled_boxes = np.array(scaled_boxes)

        # Process masks only for confident detections
        mask_coeffs = mask_coeffs[0][:, mask]  # [32, N]
        mask_protos = mask_protos[0]  # [32, 160, 160]

        # Generate masks if we have any detections
        if len(scaled_boxes) > 0:
            # Matrix multiplication and reshape
            mask_protos = mask_protos.transpose(1, 2, 0)  # [160, 160, 32]
            masks = mask_coeffs.T @ mask_protos.reshape(160*160, 32).T
            masks = masks.reshape(-1, 160, 160)
            masks = 1 / (1 + np.exp(-masks))  # sigmoid
            
            # Resize masks to original frame size
            final_masks = []
            for mask in masks:
                mask = cv2.resize(mask, (orig_shape[1], orig_shape[0]))
                final_masks.append(mask > 0.5)
            final_masks = np.array(final_masks)
        else:
            final_masks = np.zeros((0, orig_shape[0], orig_shape[1]), dtype=bool)

        # Create results object that mimics ultralytics Results
        results = DetectionResults(scaled_boxes, confidences, class_ids)

        return results, final_masks

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
            if current_time - last_process_time < 0.1:  # 10 FPS processing
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
                    # Preprocess frame
                    input_tensor, input_shape = self.preprocess_frame(frame)
                    
                    # Run inference
                    outputs = self.model([input_tensor])
                    det_output = outputs[self.output_det]
                    mask_coeffs = outputs[self.output_coeff]
                    mask_protos = outputs[self.output_proto]
                    
                    # Process outputs
                    results, masks = self.process_outputs(
                        det_output, mask_coeffs, mask_protos, frame.shape, input_shape
                    )
                    
                    if len(results.boxes) > 0:
                        # Update tracker
                        tracks = self.tracker.update(results)
                        
                        # Process tracked objects
                        for track in tracks:
                            # track format: [x1, y1, x2, y2, track_id, score, class_id]
                            box = track[:4]
                            track_id = int(track[4])
                            conf = float(track[5])
                            cls_id = int(track[6])
                            print(f"\nTrack data:")
                            print(f"box: {box}")
                            print(f"track_id: {track_id}")
                            print(f"conf: {conf}")
                            print(f"cls_id: {cls_id}")
                            print(f"len(self.class_names): {len(self.class_names)}")
                            
                            # Ensure cls_id is valid
                            if cls_id < 0 or cls_id >= len(self.class_names):
                                print(f"Warning: Invalid class ID {cls_id}")
                                continue
                                
                            cls = self.class_names[cls_id]
                            
                            if cls == self.target_class:
                                if conf > 0.5:
                                    if current_time - last_print_time > 1.0:
                                        print(f"\rTracking {cls} ID {track_id}: {conf:.2f}", end="")
                                        last_print_time = current_time
                                    
                                    # Apply temporal smoothing
                                    self.track_history[track_id].append(masks[cls_id])
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
    model_path = "openvino_models/yolo11n-seg.xml"
    try:
        detector = RealtimeSegmentation(model_path)
        detector.run()
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting steps:")
        print("1. Check if your camera is connected")
        print("2. Check if the OpenVINO model exists at:", model_path)
        print("3. Ensure you have all required dependencies installed")