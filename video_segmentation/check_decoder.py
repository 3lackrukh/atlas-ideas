#!/usr/bin/env python3
import openvino as ov

def check_decoder_structure():
    print("Loading OpenVINO Core...")
    core = ov.Core()
    
    print("\nLoading decoder model...")
    decoder_model = core.read_model("openvino_models/mobile_sam_decoder.xml")
    
    print("\nDecoder inputs:")
    for i, input in enumerate(decoder_model.inputs):
        print(f"Input {i}: {input.any_name}")
        print(f"  Shape: {input.get_partial_shape()}")
        print(f"  Type: {input.get_element_type()}")
    
    print("\nDecoder outputs:")
    for i, output in enumerate(decoder_model.outputs):
        print(f"Output {i}: {output.any_name}")
        print(f"  Shape: {output.get_partial_shape()}")
        print(f"  Type: {output.get_element_type()}")

if __name__ == "__main__":
    try:
        check_decoder_structure()
    except Exception as e:
        print(f"\nError checking decoder: {e}")
        import traceback
        traceback.print_exc()