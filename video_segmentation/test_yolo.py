#!/usr/bin/env python3
import cv2
import numpy as np
from openvino.runtime import Core
import time

def test_yolo_model(model_path, image_path=None):
    """Test OpenVINO YOLO model with enhanced debugging"""
    print("Initializing OpenVINO runtime...")
    core = Core()
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, device_name="CPU")
    
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    
    print("\nModel Details:")
    print("Input layout:", input_layer.get_partial_shape())
    print("Output layout:", output_layer.get_partial_shape())
    
    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                  'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                  'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                  'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
    def process_output(output, frame_shape, conf_threshold=0.1):
        """Process model output and return detections"""
        output = output[0]  # Get first batch
        frame_height, frame_width = frame_shape[:2]
        original_width, original_height = 640, 640  # Original YOLO input size

        # Debug: Print output statistics
        print(f"\nOutput statistics:")
        print(f"Min value: {output.min():.4f}")
        print(f"Max value: {output.max():.4f}")
        print(f"Mean value: {output.mean():.4f}")

        # Reshape predictions [84, 8400] -> [8400, 84]
        predictions = output.transpose(1, 0)

        # Debug: Print first few predictions
        print("\nFirst few raw predictions:")
        print(predictions[0, :8])

        boxes = predictions[:, :4]
        scores = predictions[:, 4:]

        # Get class predictions
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        # Debug: Print confidence statistics
        print(f"\nConfidence statistics:")
        print(f"Max confidence: {confidences.max():.4f}")
        print(f"Number of detections above {conf_threshold}: {(confidences > conf_threshold).sum()}")

        # Filter by confidence
        mask = confidences > conf_threshold
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        results = []
        for box, class_id, conf in zip(boxes, class_ids, confidences):
            x, y, w, h = box
            
            # Debug: Print raw values
            print(f"\nProcessing detection:")
            print(f"Raw box (xywh): {x:.4f}, {y:.4f}, {w:.4f}, {h:.4f}")
            
            # First normalize to 0-1 range
            x = x / original_width
            y = y / original_height
            w = w / original_width
            h = h / original_height
            
            # Then scale to frame dimensions
            x = x * frame_width
            y = y * frame_height
            w = w * frame_width
            h = h * frame_height
            
            # Convert to corners and round to integers
            x1 = int(max(0, x - w/2))
            y1 = int(max(0, y - h/2))
            x2 = int(min(frame_width, x + w/2))
            y2 = int(min(frame_height, y + h/2))
            
            # Debug: Print converted values
            print(f"Normalized coords: {x/frame_width:.4f}, {y/frame_height:.4f}, {w/frame_width:.4f}, {h/frame_height:.4f}")
            print(f"Final box (xyxy): {x1}, {y1}, {x2}, {y2}")
            print(f"Class: {class_names[class_id]} ({class_id})")
            print(f"Confidence: {conf:.4f}")
            
            if x2 > x1 and y2 > y1:
                results.append({
                    'bbox': [x1, y1, x2, y2],
                    'class_id': int(class_id),
                    'class_name': class_names[int(class_id)],
                    'conf': float(conf)
                })

        return results

    def preprocess_frame(frame):
        """Preprocess frame for model input"""
        input_height, input_width = 640, 640
        resized = cv2.resize(frame, (input_width, input_height))
        
        # Debug: Print frame statistics before and after preprocessing
        print(f"\nFrame preprocessing:")
        print(f"Original frame shape: {frame.shape}")
        print(f"Resized frame shape: {resized.shape}")
        print(f"Original frame range: [{frame.min()}, {frame.max()}]")
        
        # Convert to float and normalize
        input_image = np.array(resized, dtype=np.float32) / 255.0
        
        print(f"Normalized frame range: [{input_image.min():.4f}, {input_image.max():.4f}]")
        
        # HWC to NCHW format
        input_image = input_image.transpose((2, 0, 1))
        input_image = np.expand_dims(input_image, 0)
        
        return input_image
    
    # Initialize camera or load image
    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
    else:
        print("\nInitializing webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read from webcam")
    
    print("\nRunning inference...")
    input_tensor = preprocess_frame(frame)
    print(f"Input tensor shape: {input_tensor.shape}")
    
    start_time = time.time()
    output = compiled_model([input_tensor])[output_layer]
    inference_time = (time.time() - start_time) * 1000
    
    print(f"Inference time: {inference_time:.2f}ms")
    print(f"Output tensor shape: {output.shape}")
    
    # Process and display results
    results = process_output(output, frame.shape)
    print(f"\nDetected {len(results)} objects")
    
    # Draw detections
    for det in results:
        bbox = det['bbox']
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        label = f"{det['class_name']} {det['conf']:.2f}"
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print(f"Detected {det['class_name']} with confidence {det['conf']:.2f}")
    
    cv2.imshow('Detection Test', frame)
    print("\nPress any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if image_path is None:
        cap.release()

if __name__ == "__main__":
    try:
        model_path = "openvino_models/yolov8n.xml"
        test_yolo_model(model_path)
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()