#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from openvino.runtime import Core, serialize
import openvino as ov
import json
from typing import Tuple, Dict, Any

class ModelMetadata:
    """Stores and validates model metadata"""
    YOLO_SEG_METADATA = {
        "model_format": "YOLO11_SEG",
        "input_size": "640,640",
        "preprocessing": {
            "input_format": "BGR",
            "mean": [0, 0, 0],
            "std": [255, 255, 255],
            "input_layout": "NCHW"
        },
        "postprocessing": {
            "confidence_threshold": 0.25,
            "nms_threshold": 0.45
        },
        "output_info": {
            "boxes": "detection_output",
            "mask_coefficients": "mask_coefficients_output",
            "mask_prototypes": "mask_prototypes_output"
        }
    }

class YOLOSegWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model
    
    def forward(self, x):
        """Forward pass preserving both detection and mask outputs"""
        print(f"Input shape: {x.shape}")
        out = self.model(x)
        
        print(f"Raw output type: {type(out)}")
        if isinstance(out, (list, tuple)):
            print(f"Raw output length: {len(out)}")

            # Get detection output
            det_out = out[0]
            print(f"Detection shape: {det_out.shape}")
            det_out = det_out.permute(0, 2, 1)
            
            # Debug segmentation output structure
            print("\nInspecting segmentation outputs (out[1]):")
            for i, item in enumerate(out[1]):
                print(f"Item {i} type: {type(item)}")
                if isinstance(item, (list, tuple)):
                    print(f"Item {i} length: {len(item)}")
                    for j, subitem in enumerate(item):
                        print(f"  Subitem {j} type: {type(subitem)}")
                        if isinstance(subitem, torch.Tensor):
                            print(f"  Subitem {j} shape: {subitem.shape}")
                elif isinstance(item, torch.Tensor):
                    print(f"Item {i} shape: {item.shape}")
            
            # Extract segmentation outputs
            seg_outputs = out[1]
            # Item 1 is mask coefficients [1, 32, 8400]
            # Item 2 is prototype masks [1, 32, 160, 160]
            mask_coeffs = seg_outputs[1]  
            mask_protos = seg_outputs[2]
            
            print(f"\nExtracted shapes:")
            print(f"Mask coefficients shape: {mask_coeffs.shape}")
            print(f"Mask prototypes shape: {mask_protos.shape}")
            
            return det_out, mask_coeffs, mask_protos
        else:
            raise ValueError("Unexpected output format")

def convert_yolo11_seg(
    model_name: str = "yolo11n-seg",
    output_dir: str = "openvino_models",
    verify: bool = True
) -> Path:
    """Convert YOLOv11-seg model to OpenVINO format"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {model_name} model...")
    model = YOLO(f"{model_name}.pt"),
    model.model.eval()
    
    print("Creating wrapped model...")
    wrapped_model = YOLOSegWrapper(model)
    wrapped_model.eval()
    
    print("Testing forward pass...")
    dummy_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        det_output, mask_coeffs, mask_protos = wrapped_model(dummy_input)
        print(f"Detection output shape: {det_output.shape}")
        print(f"Mask coefficients shape: {mask_coeffs.shape}")
        print(f"Mask prototypes shape: {mask_protos.shape}")
    
    print("Exporting to ONNX...")
    onnx_path = output_dir / f"{model_name}.onnx"
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        onnx_path,
        opset_version=13,
        input_names=['images'],
        output_names=['detection_output', 'mask_coefficients_output', 'mask_prototypes_output'],
        dynamic_axes={
            'images': {0: 'batch_size'},
            'detection_output': {0: 'batch_size'},
            'mask_coefficients_output': {0: 'batch_size'},
            'mask_prototypes_output': {0: 'batch_size'}
        }
    )
    
    print("Converting to OpenVINO...")
    core = Core()
    ov_model = core.read_model(onnx_path)
    
    # Add metadata
    for key, value in ModelMetadata.YOLO_SEG_METADATA.items():
        ov_model.set_rt_info(json.dumps(value), key)
    
    ir_path = output_dir / f"{model_name}.xml"
    serialize(ov_model, ir_path)
    
    print(f"Model saved to: {ir_path}")
    return ir_path

if __name__ == "__main__":
    try:
        print("\nStarting YOLO11-seg conversion...")
        model_path = convert_yolo11_seg(verify=False)
        print(f"Model converted successfully: {model_path}")
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()