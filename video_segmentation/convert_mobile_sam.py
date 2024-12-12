#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
from mobile_sam import sam_model_registry
from openvino.runtime import Core, serialize
import openvino as ov

def convert_mobile_sam(checkpoint_path, output_dir):
    """
    Convert Mobile SAM model to OpenVINO format
    
    Args:
        checkpoint_path: Path to Mobile SAM checkpoint (mobile_sam.pt)
        output_dir: Directory to save converted models
    """
    print("Starting Mobile SAM conversion to OpenVINO...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Mobile SAM model
    print("Loading Mobile SAM model...")
    mobile_sam = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
    mobile_sam.eval()
    
    # Export image encoder
    print("Converting image encoder...")
    dummy_image = torch.randn(1, 3, 1024, 1024)
    
    # Trace encoder with error catching
    try:
        encoder_traced = torch.jit.trace(mobile_sam.image_encoder, dummy_image, strict=False)
        print("Encoder traced successfully")
    except Exception as e:
        print(f"Error tracing encoder: {e}")
        raise
    
    encoder_path = output_dir / "mobile_sam_encoder.onnx"
    try:
        torch.onnx.export(
            encoder_traced,
            dummy_image,
            encoder_path,
            opset_version=13,
            input_names=['image'],
            output_names=['image_embeddings'],
            dynamic_axes={
                'image': {2: 'height', 3: 'width'},
                'image_embeddings': {2: 'height', 3: 'width'}
            }
        )
        print("Encoder exported to ONNX successfully")
    except Exception as e:
        print(f"Error exporting encoder to ONNX: {e}")
        raise
    
    # Export mask decoder with updated wrapper
    print("\nConverting mask decoder...")
    class DecoderWrapper(torch.nn.Module):
        def __init__(self, mobile_sam):
            super().__init__()
            self.decoder = mobile_sam.mask_decoder
            self.pe_layer = mobile_sam.prompt_encoder.get_dense_pe()
            
        def forward(self, image_embeddings, point_coords, point_labels):
            # Ensure inputs are used in computation to prevent trace optimization
            batch_size = image_embeddings.shape[0]
            num_points = point_coords.shape[1]
            
            print(f"\nDebug - tensor shapes in decoder:")
            print(f"Decoder input image embeddings: {image_embeddings.shape}")
            # Get positional encoding
            image_pe = self.pe_layer.to(image_embeddings.device)
            
            # Create sparse embeddings based on points
            sparse_embeddings = torch.zeros((batch_size, num_points, image_embeddings.shape[1]), 
                                         device=image_embeddings.device)
            print(f"Created sparse embeddings: {sparse_embeddings.shape}")
            
            # Create embeddings at point locations
            for b in range(batch_size):
                for p in range(num_points):
                    # Use point_labels directly in the computation
                    label_weight = point_labels[b, p].unsqueeze(0)
                    pos = point_coords[b, p] * label_weight  # This ensures point_labels is part of computation
                    sparse_embeddings[b, p, 0:2] = pos
                    # Fill remaining embedding with label information
                    if p < sparse_embeddings.shape[2]:
                        sparse_embeddings[b, p, 2:] = label_weight.repeat(sparse_embeddings.shape[2]-2)

            # Create dense embeddings
            dense_embeddings = torch.zeros((batch_size, image_embeddings.shape[1], 64, 64),
                                           device=image_embeddings.device)
            # Run one step of decoder to see internal shapes
            test_run = self.decoder.predict_masks(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings
            )
            print(f"Decoder internal shapes: {[t.shape for t in test_run]}")
            
            # Run decoder
            masks, iou_predictions = self.decoder(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            return masks, iou_predictions
    
    # Create wrapped decoder
    wrapped_decoder = DecoderWrapper(mobile_sam)
    wrapped_decoder.eval()  # Ensure model is in eval mode
    
    # Prepare example inputs with correct shapes
    image_embeddings = torch.randn(1, 256, 64, 64)  # [batch_size, embedding_dim, height, width]
    point_coords = torch.randint(0, 1024, (1, 1, 2)).float()  # [batch_size, num_points, 2]
    point_labels = torch.randint(0, 4, (1, 1)).float()  # [batch_size, num_points]
    
    print("\nDebug info - Input shapes:")
    print(f"Image embeddings shape: {image_embeddings.shape}")
    print(f"Point coords shape: {point_coords.shape}")
    print(f"Point labels shape: {point_labels.shape}")
    
    decoder_path = output_dir / "mobile_sam_decoder.onnx"
    
    try:
        # Test forward pass before export
        with torch.no_grad():
            test_output = wrapped_decoder(image_embeddings, point_coords, point_labels)
        print("Decoder forward pass successful")
        
        # Export to ONNX with explicit input names and shapes
        torch.onnx.export(
            wrapped_decoder,
            (image_embeddings, point_coords, point_labels),
            decoder_path,
            opset_version=13,
            input_names=['image_embeddings', 'point_coords', 'point_labels'],
            output_names=['masks', 'iou_predictions'],
            dynamic_axes={
                'image_embeddings': {0: 'batch'},
                'point_coords': {0: 'batch', 1: 'points'},
                'point_labels': {0: 'batch', 1: 'points'}
            },
            do_constant_folding=False,  # Disable constant folding
            keep_initializers_as_inputs=True,  # Keep all inputs
            training=torch.onnx.TrainingMode.EVAL,  # Ensure model is in eval mode
            export_params=True
        )
        print("Decoder exported to ONNX successfully")
    except Exception as e:
        print(f"Error during decoder export: {e}")
        raise
    
    # Convert ONNX models to OpenVINO IR
    print("\nConverting ONNX models to OpenVINO IR...")
    core = Core()
    
    try:
        # Convert encoder
        print("Converting encoder to IR format...")
        ov_encoder = core.read_model(encoder_path)
        serialize(ov_encoder, output_dir / "mobile_sam_encoder.xml")
        
        # Convert decoder
        print("Converting decoder to IR format...")
        ov_decoder = core.read_model(decoder_path)
        serialize(ov_decoder, output_dir / "mobile_sam_decoder.xml")
        
        # Verify the converted models
        print("\nVerifying converted models...")
        loaded_encoder = core.read_model(output_dir / "mobile_sam_encoder.xml")
        loaded_decoder = core.read_model(output_dir / "mobile_sam_decoder.xml")
        
        # Verify encoder
        assert len(loaded_encoder.inputs) == 1, f"Encoder has {len(loaded_encoder.inputs)} inputs, expected 1"
        
        print("\nOpenVINO Decoder Input Details:")
        for idx, input in enumerate(loaded_decoder.inputs):
            print(f"Input {idx}: Name={input.get_any_name()}, Shape={input.get_partial_shape()}")

        # Verify decoder
        assert len(loaded_decoder.inputs) == 3, f"Decoder has {len(loaded_decoder.inputs)} inputs, expected 3"
        
        # Print input and output details for verification
        print("\nDecoder Input Details:")
        for i, input_node in enumerate(loaded_decoder.inputs):
            print(f"Input {i}: {input_node.any_name}, Shape: {input_node.get_partial_shape()}")
        
        print("\nDecoder Output Details:")
        for i, output_node in enumerate(loaded_decoder.outputs):
            print(f"Output {i}: {output_node.any_name}, Shape: {output_node.get_partial_shape()}")
        
        print("\nModel validation passed!")
        
    except Exception as e:
        print(f"Error during OpenVINO conversion: {e}")
        raise
    
    print("\nConversion complete!")
    print(f"Encoder saved to: {output_dir}/mobile_sam_encoder.xml")
    print(f"Decoder saved to: {output_dir}/mobile_sam_decoder.xml")
    
    return output_dir / "mobile_sam_encoder.xml", output_dir / "mobile_sam_decoder.xml"

if __name__ == "__main__":
    checkpoint_path = "mobile_sam.pt"  # Path to your Mobile SAM checkpoint
    output_dir = "openvino_models"     # Directory to save converted models
    
    try:
        encoder_path, decoder_path = convert_mobile_sam(checkpoint_path, output_dir)
        print("\nModel conversion successful!")
        print("\nYou can now use these models in your realtime segmentation code:")
        print(f"encoder_path = \"{encoder_path}\"")
        print(f"decoder_path = \"{decoder_path}\"")
    except Exception as e:
        print(f"\nError during conversion: {e}")
        print("\nDebug information:")
        print("1. Check if the checkpoint file exists and is valid")
        print("2. Ensure you have enough disk space")
        print("3. Verify you have the correct versions of torch and openvino installed")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()