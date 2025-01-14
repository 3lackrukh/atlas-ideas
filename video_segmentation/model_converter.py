#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from mobile_sam import sam_model_registry
from openvino.runtime import Core, serialize
import openvino as ov
import json
from typing import Tuple, Dict, Any
import cv2
import gc

class ModelMetadata:
    """Stores and validates model metadata"""
    YOLO_METADATA = {
        "model_format": "YOLO_V8",
        "input_size": "640,640",
        "num_classes": 80,
        "preprocessing": {
            "input_format": "BGR",
            "mean": [0, 0, 0],
            "std": [255, 255, 255],
            "input_layout": "NCHW",
            "resize_mode": "letterbox"
        },
        "postprocessing": {
            "output_format": "batched_predictions",
            "confidence_threshold": 0.25,
            "nms_threshold": 0.45
        }
    }
    
    SAM_METADATA = {
        "model_format": "Mobile_SAM",
        "input_size": "1024,1024",
        "preprocessing": {
            "input_format": "RGB",
            "mean": [0, 0, 0],
            "std": [255, 255, 255],
            "input_layout": "NCHW",
            "resize_mode": "centered_pad"
        },
        "prompt_format": {
            "point_format": "normalized",
            "label_format": "binary"
        }
    }

class YOLOWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model
    
    def forward(self, x):
        print(f"Input shape: {x.shape}")
        out = self.model(x)
        
        print(f"Raw output type: {type(out)}")
        if isinstance(out, (list, tuple)):
            print(f"Raw output length: {len(out)}")
            print(f"First element shape: {out[0].shape}")
            out = out[0]
        else:
            print(f"Raw output shape: {out.shape}")
        
        out = out.permute(0, 2, 1)
        print(f"Final output shape: {out.shape}")
        return out

def convert_yolov8(
    model_name: str = "yolov8n",
    output_dir: str = "openvino_models",
    verify: bool = True
) -> Path:
    """Convert YOLOv8 model to OpenVINO format"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {model_name} model...")
    model = YOLO(f"{model_name}.pt")
    model.model.eval()
    
    print("Creating wrapped model...")
    wrapped_model = YOLOWrapper(model)
    wrapped_model.eval()
    
    print("Testing forward pass...")
    dummy_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        test_output = wrapped_model(dummy_input)
        print(f"Final test output shape: {test_output.shape}")
    
    print("Exporting to ONNX...")
    onnx_path = output_dir / f"{model_name}.onnx"
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        onnx_path,
        opset_version=13,
        input_names=['images'],
        output_names=['predictions'],
        dynamic_axes={
            'images': {0: 'batch_size'},
            'predictions': {0: 'batch_size'}
        }
    )
    
    print("Converting to OpenVINO...")
    core = Core()
    ov_model = core.read_model(onnx_path)
    
    for key, value in ModelMetadata.YOLO_METADATA.items():
        ov_model.set_rt_info(json.dumps(value), key)
    
    ir_path = output_dir / f"{model_name}.xml"
    serialize(ov_model, ir_path)
    
    print(f"Model saved to: {ir_path}")
    return ir_path

def convert_mobile_sam(
    checkpoint_path: str,
    output_dir: str = "openvino_models",
    verify: bool = True
) -> Tuple[Path, Path]:
    """Convert Mobile SAM model to OpenVINO format with memory optimization"""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading Mobile SAM from: {checkpoint_path}")
    
    # Clear memory before loading model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    try:
        mobile_sam = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
        mobile_sam.eval()
        
        print("\nConverting encoder...")
        encoder_path = output_dir / "mobile_sam_encoder.onnx"
        
        print("Testing encoder...")
        with torch.no_grad():
            dummy_image = torch.randn(1, 3, 1024, 1024)
            test_embedding = mobile_sam.image_encoder(dummy_image)
            print(f"Test encoder output shape: {test_embedding.shape}")
            del test_embedding
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print("Exporting encoder to ONNX...")
        torch.onnx.export(
            mobile_sam.image_encoder,
            dummy_image,
            encoder_path,
            opset_version=13,
            input_names=['image'],
            output_names=['image_embeddings'],
            dynamic_axes={
                'image': {0: 'batch'},
                'image_embeddings': {0: 'batch'}
            }
        )
        
        del dummy_image
        gc.collect()
        
        print("\nConverting encoder to OpenVINO...")
        core = Core()
        ov_encoder = core.read_model(encoder_path)
        encoder_ir_path = output_dir / "mobile_sam_encoder.xml"
        serialize(ov_encoder, encoder_ir_path)
        
        print("\nEncoder conversion successful!")
        
        # Convert decoder with fixed structure
        print("\nConverting decoder...")
        
        class DecoderWrapper(torch.nn.Module):
            def __init__(self, mobile_sam):
                super().__init__()
                self.decoder = mobile_sam.mask_decoder
                self.pe = mobile_sam.prompt_encoder.get_dense_pe()

            def forward(self, image_embeddings, point_coords, point_labels):
                """
                Args:
                    image_embeddings: [B, 256, 64, 64]
                    point_coords: [B, N, 2]
                    point_labels: [B, N] - this input was missing before
                """
                # Create sparse embeddings using both coords and labels
                sparse_embeddings = torch.zeros(
                    (image_embeddings.shape[0], point_coords.shape[1], 256),
                    device=image_embeddings.device
                )

                # For each point, incorporate its label into the embedding
                for b in range(image_embeddings.shape[0]):  # batch dimension
                    for p in range(point_coords.shape[1]):  # points dimension
                        sparse_embeddings[b, p, 0:2] = point_coords[b, p]
                        # Use the label information
                        sparse_embeddings[b, p, 2] = point_labels[b, p]

                # Create dense embeddings
                dense_embeddings = torch.zeros(
                    (image_embeddings.shape[0], 256, 64, 64),
                    device=image_embeddings.device
                )

                # Get positional encoding
                image_pe = self.pe.to(image_embeddings.device)

                # Run decoder with modified sparse embeddings
                masks, iou_predictions = self.decoder.predict_masks(
                    image_embeddings=image_embeddings,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings
                )

                return masks[:, :1], iou_predictions[:, :1]
        
        print("Creating decoder wrapper...")
        wrapped_decoder = DecoderWrapper(mobile_sam)
        decoder_path = output_dir / "mobile_sam_decoder.onnx"
        
        print("Testing decoder...")
        with torch.no_grad():
            # Create inputs once and reuse them
            dummy_embedding = torch.randn(1, 256, 64, 64)
            dummy_points = torch.randint(0, 1024, (1, 1, 2)).float()
            dummy_labels = torch.ones(1, 1)
    
            test_masks, test_iou = wrapped_decoder(dummy_embedding, dummy_points, dummy_labels)
            print(f"Test decoder output shapes - Masks: {test_masks.shape}, IoU: {test_iou.shape}")
    
            # Don't delete these since we need them for export
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print("Exporting decoder to ONNX...")
        # Use the same inputs from testing - remove the redundant creation here
        torch.onnx.export(
            wrapped_decoder,
            (dummy_embedding, dummy_points, dummy_labels),  # Reuse the same inputs
            decoder_path,
            opset_version=13,
            input_names=['image_embeddings', 'point_coords', 'point_labels'],
            output_names=['masks', 'iou_predictions'],
            dynamic_axes={
                'image_embeddings': {0: 'batch'},
                'point_coords': {0: 'batch', 1: 'num_points'},
                'point_labels': {0: 'batch', 1: 'num_points'}
            }
        )
        
        print("Converting decoder to OpenVINO...")
        ov_decoder = core.read_model(decoder_path)
        decoder_ir_path = output_dir / "mobile_sam_decoder.xml"
        serialize(ov_decoder, decoder_ir_path)
        
        print("\nDecoder conversion successful!")
        return encoder_ir_path, decoder_ir_path
        
    finally:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

if __name__ == "__main__":
    # Convert YOLO
    try:
        print("\nStarting YOLO conversion...")
        yolo_path = convert_yolov8(verify=False)
        print(f"YOLO model converted successfully: {yolo_path}")
    except Exception as e:
        print(f"YOLO conversion failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Convert SAM
    try:
        print("\nStarting SAM conversion...")
        if Path("mobile_sam.pt").exists():
            encoder_path, decoder_path = convert_mobile_sam("mobile_sam.pt", verify=False)
            print(f"SAM models converted successfully:")
            print(f"Encoder: {encoder_path}")
            print(f"Decoder: {decoder_path}")
        else:
            print("mobile_sam.pt not found. Skipping SAM conversion.")
    except Exception as e:
        print(f"SAM conversion failed: {e}")
        import traceback
        traceback.print_exc()