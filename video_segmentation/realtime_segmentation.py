#!/usr/bin/env python3
import cv2
import speech_recognition as sr
import threading
from queue import Queue, Empty
import numpy as np
from collections import deque
import time
import openvino as ov

class RealtimeSegmentation:
    def __init__(self, sam_encoder_path, sam_decoder_path):
        self.device = "CPU"
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
        
        # Initialize OpenVINO runtime
        print("Loading models...")
        core = ov.Core()
        
        # Load YOLO model
        self.yolo_model = core.compile_model("yolov8n.xml", device_name=self.device)
        self.yolo_input_layer = self.yolo_model.input(0)
        self.yolo_output_layer = self.yolo_model.output(0)
        print("YOLO Model Details:")
        print(f"Input layer shape: {self.yolo_input_layer.shape}")
        print(f"Output layer shape: {self.yolo_output_layer.shape}")
        
        # Load Mobile SAM OpenVINO models
        self.sam_encoder = core.compile_model(sam_encoder_path, device_name=self.device)
        self.sam_decoder = core.compile_model(sam_decoder_path, device_name=self.device)
        
        # Validate models
        self.validate_models()
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        
        self.command_queue = Queue()
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue()
        self.last_masks = deque(maxlen=2)
        self.tracking_active = False
        self.target_class = None
    
    def validate_models(self):
        """Validate loaded models have correct input/output configuration"""
        # Validate encoder
        assert len(self.sam_encoder.inputs) == 1, "Encoder should have 1 input"
        assert len(self.sam_encoder.outputs) == 1, "Encoder should have 1 output"
        
        # Validate decoder
        assert len(self.sam_decoder.inputs) == 3, "Decoder should have 3 inputs"
        assert len(self.sam_decoder.outputs) == 2, "Decoder should have 2 outputs"
        
        print("Model validation passed!")
        
        # Print decoder input/output details
        print("\nSAM Decoder Info:")
        print("Input details:")
        for input in self.sam_decoder.inputs:
            print(f"- {input.any_name}: {input.get_partial_shape()}")
        print("\nOutput details:")
        for output in self.sam_decoder.outputs:
            print(f"- {output.any_name}: {output.get_partial_shape()}")

    def verify_input_shapes(self, image_embeddings, point_coords, point_labels):
        """Verify shapes of inputs before running decoder"""
        print(f"Image embeddings shape: {image_embeddings.shape}")  # Should be [1, 256, 64, 64]
        print(f"Point coords shape: {point_coords.shape}")          # Should be [1, 1, 2]
        print(f"Point labels shape: {point_labels.shape}")          # Should be [1, 1]
        
        assert image_embeddings.shape == (1, 256, 64, 64), f"Unexpected embeddings shape: {image_embeddings.shape}"
        assert point_coords.shape[0:2] == (1, 1), f"Unexpected coords shape: {point_coords.shape}"
        assert point_labels.shape == (1, 1), f"Unexpected labels shape: {point_labels.shape}"

    def preprocess_sam_image(self, frame):
        """Preprocess image for Mobile SAM"""
        # Resize to 1024x1024
        input_size = 1024
        h, w = frame.shape[:2]
        scale = input_size / max(h, w)
        scaled_h, scaled_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(frame, (scaled_w, scaled_h))
        
        # Create padding
        pad_h = input_size - scaled_h
        pad_w = input_size - scaled_w
        top = pad_h // 2
        left = pad_w // 2
        
        padded = cv2.copyMakeBorder(resized, top, pad_h - top, left, pad_w - left,
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        # Normalize and convert to NCHW format
        input_image = padded.astype(np.float32) / 255.0
        input_image = np.transpose(input_image, (2, 0, 1))[None, ...]
        
        return input_image, (scale, top, left)

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

        def preprocess_yolo_image(frame):
            input_height, input_width = 640, 640
            resized = cv2.resize(frame, (input_width, input_height))
            input_image = np.array(resized, dtype=np.float32) / 255.0
            input_image = input_image.transpose((2, 0, 1))
            input_image = np.expand_dims(input_image, 0)
            return input_image

        def process_yolo_output(output, frame_shape, conf_threshold=0.2):
            output = output[0]
            boxes = output[0:4, :].transpose()
            confidences = output[4, :]
            class_scores = output[5:, :].transpose()

            results = []
            for i in range(len(confidences)):
                if confidences[i] > conf_threshold:
                    class_id = np.argmax(class_scores[i])
                    score = float(class_scores[i][class_id])
                    
                    if score > 0.001:
                        x, y, w, h = boxes[i]
                        x1 = int((x - w/2) * frame_shape[1])
                        y1 = int((y - h/2) * frame_shape[0])
                        x2 = int((x + w/2) * frame_shape[1])
                        y2 = int((y + h/2) * frame_shape[0])
                        
                        results.append({
                            'bbox': [x1, y1, x2, y2],
                            'conf': float(confidences[i] * score),
                            'class_id': class_id
                        })

            return results

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
                    # Run YOLO inference
                    input_tensor = preprocess_yolo_image(frame)
                    results = self.yolo_model([input_tensor])[self.yolo_output_layer]
                    detections = process_yolo_output(results, frame.shape)
                    
                    for det in detections:
                        class_name = self.class_names[det['class_id']]
                        
                        if class_name == self.target_class:
                            if current_time - last_print_time > 1.0:
                                print(f"\rTracking {class_name}: {det['conf']:.2f}", end="")
                                last_print_time = current_time
                            
                            # Get center point for SAM
                            bbox = det['bbox']
                            center_x = int((bbox[0] + bbox[2]) / 2)
                            center_y = int((bbox[1] + bbox[3]) / 2)
                            
                            # Prepare image for SAM
                            sam_input, (scale, top, left) = self.preprocess_sam_image(frame)
                            
                            # Run SAM encoder
                            image_embeddings = self.sam_encoder([sam_input])[0]
                            
                            # Prepare point coordinates
                            point_coords = np.array([[[center_x * scale + left, center_y * scale + top]]], dtype=np.float32)
                            point_labels = np.array([[1]], dtype=np.float32)
                            
                            # Verify shapes before inference
                            self.verify_input_shapes(image_embeddings, point_coords, point_labels)
                            
                            # Run SAM decoder with all inputs
                            decoder_outputs = self.sam_decoder([
                                image_embeddings,
                                point_coords,
                                point_labels
                            ])
                            
                            # Get masks and predictions
                            masks = decoder_outputs[0]
                            iou_predictions = decoder_outputs[1]
                            
                            # Get single mask with highest IoU
                            best_mask_idx = np.argmax(iou_predictions[0])
                            mask = masks[0, best_mask_idx]
                            
                            # Resize mask to original image size
                            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0])) > 0.5
                            
                            # Smooth mask using temporal averaging
                            self.last_masks.append(mask)
                            smooth_mask = np.mean(self.last_masks, axis=0) > 0.5
                            
                            self.result_queue.put({
                                'bbox': bbox,
                                'mask': smooth_mask,
                                'class': class_name,
                                'conf': det['conf']
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
    sam_encoder_path = "openvino_models/mobile_sam_encoder.xml"
    sam_decoder_path = "openvino_models/mobile_sam_decoder.xml"
    
    try:
        detector = RealtimeSegmentation(sam_encoder_path, sam_decoder_path)
        detector.run()
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting steps:")
        print("1. Check if your camera is connected")
        print("2. Run 'ls -l /dev/video*' to list available cameras")
        print("3. Ensure you have proper permissions: 'sudo usermod -a -G video $USER'")
        print("4. Try rebooting if needed")