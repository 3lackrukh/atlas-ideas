#!/usr/bin/env python3
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import openvino as ov

def load_models():
    """Load both PyTorch and OpenVINO models"""
    print("Loading models...")
    
    # Load PyTorch model
    torch_model = YOLO('yolo11n-seg.pt')
    torch_model.model.eval()
    print("PyTorch model loaded")
    
    # Load OpenVINO model
    core = ov.Core()
    ov_model = core.compile_model("openvino_models/yolo11n-seg.xml")
    print("OpenVINO model loaded")
    
    return torch_model, ov_model

def prepare_input(image_path=None):
    """Prepare input tensor. If no image provided, use random data"""
    if image_path:
        # Load and preprocess real image
        frame = cv2.imread(image_path)
        frame = cv2.resize(frame, (640, 640))
    else:
        # Create random test image
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Normalize and convert to proper format
    input_tensor = frame.astype(np.float32) / 255.0
    input_tensor = input_tensor.transpose(2, 0, 1)
    input_tensor = np.expand_dims(input_tensor, 0)
    
    return frame, input_tensor

def inspect_segmentation_process(torch_outputs, ov_outputs):
    """Inspect how segmentation tensors are used"""
    print("\nInspecting Segmentation Process:")
    
    print("\nPyTorch segmentation tensors:")
    seg_outputs = torch_outputs[1]
    
    # Mask coefficients
    coeffs = seg_outputs[1]
    print("\nMask coefficients:")
    print(f"  Shape: {coeffs.shape}")
    print(f"  Value range: {coeffs.min().item():.4f} to {coeffs.max().item():.4f}")
    print(f"  Mean value: {coeffs.mean().item():.4f}")
    
    # Prototype masks
    protos = seg_outputs[2]
    print("\nPrototype masks:")
    print(f"  Shape: {protos.shape}")
    print(f"  Value range: {protos.min().item():.4f} to {protos.max().item():.4f}")
    print(f"  Mean value: {protos.mean().item():.4f}")
    
    print("\nOpenVINO segmentation tensors:")
    print("\nMask coefficients:")
    ov_coeffs = ov_outputs['mask_coefficients_output']
    print(f"  Shape: {ov_coeffs.shape}")
    print(f"  Value range: {ov_coeffs.min():.4f} to {ov_coeffs.max():.4f}")
    print(f"  Mean value: {ov_coeffs.mean():.4f}")
    
    print("\nPrototype masks:")
    ov_protos = ov_outputs['mask_prototypes_output']
    print(f"  Shape: {ov_protos.shape}")
    print(f"  Value range: {ov_protos.min():.4f} to {ov_protos.max():.4f}")
    print(f"  Mean value: {ov_protos.mean():.4f}")
    
def validate_mask_outputs(torch_coeffs, torch_protos, ov_coeffs, ov_protos):
    """Validate mask-related outputs between PyTorch and OpenVINO"""
    print("\nValidating mask outputs:")
    print(f"PyTorch coefficients: {torch_coeffs.shape}, range: [{torch_coeffs.min():.4f}, {torch_coeffs.max():.4f}]")
    print(f"PyTorch prototypes: {torch_protos.shape}, range: [{torch_protos.min():.4f}, {torch_protos.max():.4f}]")
    print(f"OpenVINO coefficients: {ov_coeffs.shape}, range: [{ov_coeffs.min():.4f}, {ov_coeffs.max():.4f}]")
    print(f"OpenVINO prototypes: {ov_protos.shape}, range: [{ov_protos.min():.4f}, {ov_protos.max():.4f}]")
    
    # Compare numerical differences
    coeff_diff = np.abs(torch_coeffs - ov_coeffs).max()
    proto_diff = np.abs(torch_protos - ov_protos).max()
    print(f"\nMax differences:")
    print(f"Coefficients: {coeff_diff:.6f}")
    print(f"Prototypes: {proto_diff:.6f}")

def compare_outputs(torch_outputs, ov_outputs, frame):
    """Compare outputs from both models"""
    print("\nComparing outputs:")
    
    # Get outputs
    det_out, seg_out = torch_outputs
    
    # 1. Raw shape comparison
    print("\nRaw shapes:")
    print(f"PyTorch detection: {det_out.shape}")
    if isinstance(seg_out, (tuple, list)):
        print("PyTorch segmentation:")
        for i, item in enumerate(seg_out):
            if isinstance(item, torch.Tensor):
                print(f"  Tensor {i} shape: {item.shape}")
    
    print("\nOpenVINO shapes:")
    for output in ov_outputs:
        print(f"  {output}: {ov_outputs[output].shape}")
    
    # 2. Detection output comparison
    print("\nDetection output comparison:")
    # Convert PyTorch detection output to match OpenVINO's ordering
    torch_det = det_out.permute(0, 2, 1).cpu().numpy()  # [1, 8400, 116]
    ov_det = ov_outputs['detection_output']  # [1, 8400, 116]
    
    print(f"PyTorch shape (after transpose): {torch_det.shape}")
    print(f"OpenVINO shape: {ov_det.shape}")
    
    # Compare first few detections
    print("\nFirst 5 detections, first 5 values each:")
    for i in range(5):
        print(f"\nDetection {i}:")
        print(f"  PyTorch:  {torch_det[0, i, :5]}")
        print(f"  OpenVINO: {ov_det[0, i, :5]}")
    
    # Calculate detection differences
    det_diff = np.abs(torch_det - ov_det).max()
    det_mean_diff = np.mean(np.abs(torch_det - ov_det))
    print(f"\nDetection differences:")
    print(f"  Max difference: {det_diff:.6f}")
    print(f"  Mean difference: {det_mean_diff:.6f}")
    
    # 3. Validate mask outputs
    torch_coeffs = torch_outputs[1][1].cpu().numpy()
    torch_protos = torch_outputs[1][2].cpu().numpy()
    ov_coeffs = ov_outputs['mask_coefficients_output']
    ov_protos = ov_outputs['mask_prototypes_output']
    
    validate_mask_outputs(torch_coeffs, torch_protos, ov_coeffs, ov_protos)
    inspect_segmentation_process(torch_outputs, ov_outputs)
    
    cv2.imshow("Input Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def test_models():
    """Main test function"""
    try:
        # Load models
        torch_model, ov_model = load_models()
        
        # Prepare input
        frame, input_tensor = prepare_input()
        
        print("\nRunning inference...")
        # PyTorch inference
        with torch.no_grad():
            # Get raw model outputs instead of processed Results
            torch_outputs = torch_model.model(torch.from_numpy(input_tensor))
            print("\nRaw PyTorch outputs:")
            for i, out in enumerate(torch_outputs):
                if isinstance(out, torch.Tensor):
                    print(f"Output {i} shape:", out.shape)
                elif isinstance(out, (tuple, list)):
                    print(f"Output {i} (tuple/list):")
                    for j, item in enumerate(out):
                        if isinstance(item, torch.Tensor):
                            print(f"  Item {j} shape:", item.shape)
        print("PyTorch inference complete")
        
        # OpenVINO inference
        ov_outputs = ov_model([input_tensor])
        print("OpenVINO inference complete")
        
        # Compare outputs
        compare_outputs(torch_outputs, ov_outputs, frame)
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_models()