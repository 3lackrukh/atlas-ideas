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
import types
from typing import Optional, List, Dict, Any

class DetectionResults:
    """
    Container class for detection results that mimics ultralytics Results format
    for compatibility with BYTETracker
    """
    def __init__(self, boxes, scores, cls):
        self.boxes = boxes
        self.conf = scores
        self.cls = cls
        self.xywh = boxes  # BYTETracker will use this if xywhr is not present

class RealtimeSegmentation:
    def __init__(self, model_path: str):
        print("Starting initialization...")

        try:
            # Initialize OpenVINO
            core = ov.Core()
            self.device = "GPU" if "GPU" in core.available_devices else "CPU"
            print(f"Using device: {self.device}")
            
            # Create a YOLO model for streaming
            self.model = core.compile_model(model_path, device_name=self.device)
            
            # Get input and output nodes
            self.input_layer = self.model.input(0)
            self.output_det = self.model.output("detection_output")
            self.output_coeff = self.model.output("mask_coefficients_output")
            self.output_proto = self.model.output("mask_prototypes_output")
            
            # Configure streaming attributes
            self.batch_size = 1  # Can be increased if hardware allows
            self.stride = 1  # Process every frame, adjust if needed
            self.stream_buffer = True  # Enable frame buffering for smooth output
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenVINO model: {e}")

        # Initialize tracker with streaming-optimized parameters
        tracker_args = types.SimpleNamespace(
            track_buffer=30,
            track_high_thresh=0.6,
            track_low_thresh=0.1,
            new_track_thresh=0.7,
            match_thresh=0.8,
            fuse_score=True,
            proximity_thresh=0.6,
            appearance_thresh=0.3,
            with_reid=False,  # Disabled for streaming performance
            gmc_method='sparseOptFlow'
        )
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
        
        # Initialize video stream properties
        self.frame_source = None
        self.stream_active = False
        self.tracking_classes = set()
        self.active_trackers = {}

        # Initialize queues with appropriate buffer sizes
        self.frame_queue = Queue(maxsize=2)  # Small buffer for real-time processing
        self.result_queue = Queue(maxsize=2)

        # Performance monitoring
        self.fps_tracker = deque(maxlen=30)
        self.last_fps_print = time.time()

    def setup_video_stream(self, source=0):
        """Configure video stream for optimal performance"""
        cap = cv2.VideoCapture(source)

        # Optimize camera buffer
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Set reasonable resolution for real-time processing
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Enable camera properties for better streaming
        if hasattr(cv2, 'CAP_PROP_HW_ACCELERATION'):
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, 1)  # Enable hardware acceleration if available

        return cap


    def setup_speech_recognition(self):
        """Initialize speech recognition with error handling"""
        try:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 4000  # Adjusted for better sensitivity
            self.recognizer.dyna_energy_threshold = True
            self.mic = sr.Microphone()
            
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize speech recognition: {e}")

    def preprocess_frame(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """
        Preprocess frame for model input
        
        Args:
            frame (np.ndarray): Input frame from camera
            
        Returns:
            tuple: Processed input tensor and input shape
        """
        input_height, input_width = 640, 640
        resized = cv2.resize(frame, (input_width, input_height))
        input_image = np.array(resized, dtype=np.float32) / 255.0
        input_image = input_image.transpose((2, 0, 1))
        input_image = np.expand_dims(input_image, 0)
        return input_image, (input_height, input_width)

    def process_outputs(self, det_output: np.ndarray, mask_coeffs: np.ndarray,
                       mask_protos: np.ndarray, orig_shape: tuple,
                       input_shape: tuple) -> tuple[DetectionResults, np.ndarray]:
        """
        Process model outputs into detections and masks
        
        Args:
            det_output (np.ndarray): Detection output from model
            mask_coeffs (np.ndarray): Mask coefficients from model
            mask_protos (np.ndarray): Mask prototypes from model
            orig_shape (tuple): Original frame shape
            input_shape (tuple): Input shape used for model
            
        Returns:
            tuple: DetectionResults object and processed masks
        """
        # Process detection outputs
        predictions = det_output[0]  # Shape: [8400, 116]
        boxes = predictions[:, :4]  # [8400, 4]
        scores = predictions[:, 4:]  # [8400, 112]

        # Get predictions and filter by confidence
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

        results = DetectionResults(scaled_boxes, confidences, class_ids)
        return results, final_masks

    def process_stream(self, frame):
        """Process a single frame in streaming mode"""
        if not self.tracking_classes:
            return None
            
        try:
            # Prepare frame for inference
            input_tensor, input_shape = self.preprocess_frame(frame)
            
            # Run inference with streaming optimization
            outputs = self.model([input_tensor])
            results, masks = self.process_outputs(
                outputs[self.output_det],
                outputs[self.output_coeff],
                outputs[self.output_proto],
                frame.shape,
                input_shape
            )
            
            if len(results.boxes) > 0:
                # Update tracker with streaming results
                tracks = self.tracker.update(results)
                
                # Process tracked objects efficiently
                frame_results = []
                for track in tracks:
                    cls_id = int(track[6])
                    if cls_id < 0 or cls_id >= len(self.class_names):
                        continue
                        
                    cls = self.class_names[cls_id]
                    if cls in self.tracking_classes and track[5] > 0.5:  # Confidence threshold
                        track_id = int(track[4])
                        
                        # Efficient mask smoothing
                        if track_id not in self.active_trackers:
                            self.active_trackers[track_id] = deque(maxlen=3)
                        
                        self.active_trackers[track_id].append(masks[cls_id])
                        smooth_mask = np.mean(list(self.active_trackers[track_id]), axis=0) > 0.5
                        
                        frame_results.append({
                            'bbox': track[:4].tolist(),
                            'mask': smooth_mask,
                            'class': cls,
                            'conf': float(track[5]),
                            'track_id': track_id
                        })
                
                return frame_results
                
        except Exception as e:
            print(f"Stream processing error: {e}")
            return None

    def speech_worker(self):
        """Listen for speech commands continuously"""
        print("Starting speech recognition...")
        
        while True:
            try:
                with self.mic as source:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                    text = self.recognizer.recognize_google(audio)
                    
                    if text:
                        text = text.lower()
                        if "track" in text:
                            target = text.replace("track", "").strip()
                            if target in self.class_names:
                                self.tracking_classes.add(target)
                                print(f"\nNow tracking: {target}")
                        elif "stop tracking" in text:
                            target = text.replace("stop tracking", "").strip()
                            if target in self.tracking_classes:
                                self.tracking_classes.remove(target)
                                print(f"\nStopped tracking: {target}")
                        elif "stop all" in text:
                            self.tracking_classes.clear()
                            print("\nStopped all tracking")
                            
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                if not isinstance(e, sr.UnknownValueError):
                    print(f"Speech recognition error: {e}")
                continue

    def find_camera(self) -> Optional[cv2.VideoCapture]:
        """
        Try different camera backends and indices
        
        Returns:
            Optional[cv2.VideoCapture]: Camera object if found, None otherwise
        """
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

    def visualize_results(self, frame: np.ndarray,
                         results: List[Dict[str, Any]]) -> np.ndarray:
        """
        Enhanced visualization with multiple tracked objects
        
        Args:
            frame (np.ndarray): Input frame
            results (List[Dict]): List of detection results
            
        Returns:
            np.ndarray: Annotated frame
        """
        if not results:
            return frame
            
        # Create separate overlays for each tracked object
        overlay = frame.copy()
        
        for result in results:
            # Apply mask with unique colors based on track_id
            if result['mask'] is not None:
                color = self.get_tracking_color(result['track_id'])
                overlay[result['mask']] = overlay[result['mask']] * color
            
            # Draw bounding box
            bbox = result['bbox']
            cv2.rectangle(frame,
                         (int(bbox[0]), int(bbox[1])),
                         (int(bbox[2]), int(bbox[3])),
                         color,
                         2)
            
            # Add label
            label = f"{result['class']} {result['conf']:.2f} ID:{result['track_id']}"
            cv2.putText(frame,
                       label,
                       (int(bbox[0]), int(bbox[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.6,
                       color,
                       2)
        
        # Blend overlay with original frame
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        return frame

    def get_tracking_color(self, track_id: int) -> tuple[float, float, float]:
        """
        Generate consistent color for each track ID
        
        Args:
            track_id (int): Tracking ID number
            
        Returns:
            tuple: RGB color values as floats
        """
        colors = [
            (0.3, 1.0, 0.3),  # Green
            (0.3, 0.3, 1.0),  # Blue
            (1.0, 0.3, 0.3),  # Red
            (0.3, 1.0, 1.0),  # Cyan
            (1.0, 0.3, 1.0),  # Magenta
            (1.0, 1.0, 0.3)   # Yellow
        ]
        return colors[track_id % len(colors)]

    def run(self):
        """Main execution loop optimized for streaming"""
        print("Initializing video stream...")
        cap = self.setup_video_stream()
        if not cap.isOpened():
            raise RuntimeError("Failed to open video stream")

        cv2.namedWindow('Detection and Segmentation', cv2.WINDOW_NORMAL)

        print("Starting stream processing...")
        # Start speech recognition thread
        speech_thread = threading.Thread(target=self.speech_worker, daemon=True)
        speech_thread.start()
        print("Speech recognition started")
        last_frame_time = time.time()
        frame_count = 0

        try:
            while True:
                ret = cap.grab()  # Efficient frame grabbing
                if not ret:
                    continue
                    
                # Only retrieve and process frame if enough time has passed
                current_time = time.time()
                if current_time - last_frame_time >= 1.0 / 30.0:  # Target 30 FPS
                    ret, frame = cap.retrieve()
                    if not ret:
                        continue
                    
                    # Process frame
                    results = self.process_stream(frame)
                    if results:
                        display_frame = self.visualize_results(frame.copy(), results)
                    else:
                        display_frame = frame
                    
                    # Update FPS tracking
                    frame_count += 1
                    if current_time - self.last_fps_print >= 1.0:
                        fps = frame_count / (current_time - self.last_fps_print)
                        print(f"\rProcessing at {fps:.1f} FPS", end="")
                        frame_count = 0
                        self.last_fps_print = current_time
                    
                    # Display frame
                    cv2.imshow('Detection and Segmentation', display_frame)
                    last_frame_time = current_time
                
                # Check for exit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Stream error: {e}")
        finally:
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
        print("4. Check system permissions for camera access")
        print("5. Verify OpenVINO installation and device compatibility")