import torch
import torch.nn as nn
import torch.onnx
import sys
import os
import onnx
import onnx.helper
import onnx.mapping
import numpy as np
import argparse
import time
import io # For saving ONNX model to a byte stream in memory for verification
sys.path.append(os.getcwd())

class ONNXConverter:
    """
    A class to handle the entire process of converting a PyTorch model to ONNX,
    and then modifying the ONNX graph for chunky (HWC) RGBA input/output and
    optimized input data type handling.
    """
    def __init__(self, pytorch_model_path, output_onnx_path):
        self.pytorch_model_path = pytorch_model_path
        self.output_onnx_path = output_onnx_path
        self.model = None
        self.device = None

    def load_pytorch_model(self):
        """
        Loads the PyTorch model from the specified path and prepares it for export.
        """
        print(f"Loading PyTorch model from {self.pytorch_model_path}...")
        try:
            # Use map_location to ensure CPU loading if current device is CPU, then move.
            # This avoids CUDA OOM if the model is large and was saved on GPU.
            self.model = torch.load(os.path.abspath(self.pytorch_model_path), map_location='cpu', weights_only=False)
            print("PyTorch model loaded successfully.")
        except Exception as e:
            print(f"Error loading PyTorch model from {self.pytorch_model_path}: {e}")
            print("Please ensure the file exists and is a valid PyTorch model saved with torch.save(model, ...).")
            sys.exit(1)

        self.model.eval()
        print("Model set to evaluation mode.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model moved to device: {self.device}")

        # Convert to half precision only if running on CUDA and model is not already half
        if self.device.type == 'cuda' and next(self.model.parameters()).dtype != torch.float16:
            self.model.half()
            print("Model parameters converted to Half precision (FP16) for export.")
        elif self.device.type == 'cuda' and next(self.model.parameters()).dtype == torch.float16:
            print("Model parameters already in Half precision (FP16).")
        else:
            print("Running on CPU, skipping FP16 conversion for model parameters.")

        if hasattr(self.model, 'fuse_layers') and callable(self.model.fuse_layers):
            print("Calling fuse_layers method for optimization...")
            self.model.eval() # Redundant call, but harmless
            self.model.fuse_layers()
            print("Layer fusion complete.")
        else:
            print("Warning: Model instance does not have a 'fuse_layers' method or it's not callable.")
            print("Skipping layer fusion.")

    def export_to_onnx_in_memory(self):
        """
        Exports the PyTorch model to ONNX format and returns the ONNX model object in memory.
        """
        # Original dummy input, assuming RGBA (4 channels)
        dummy_input = torch.randint(0, 256, (1, 4, 576, 752), dtype=torch.uint8).to(self.device)
        print(f"Created dummy input tensor with shape {dummy_input.shape} and dtype {dummy_input.dtype} on device {self.device}")

        input_names = ["input_rgba_uint8"]
        output_names = ["output_rgba_float16_scaled"]
        dynamic_axes = {}

        print("Exporting model to ONNX format in memory...")
        try:
            f = io.BytesIO() # Use a BytesIO object as a file-like object
            torch.onnx.export(
                self.model,
                dummy_input,
                f, # Export to in-memory buffer
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=13,
                verbose=False,
                input_dtypes=[torch.uint8], # Specify input dtype for ONNX graph
                # output_dtypes cannot be specified directly to FP16 if the model output is not directly FP16.
                # The model's last layer output will be FP16 if model.half() is called.
                # The final scaling to 255.0 happens and maintains FP16.
                # Let PyTorch determine the output dtype. We will enforce it later in modification.
            )
            print("Model exported successfully to ONNX format in memory!")

            # Load the ONNX model from the in-memory buffer
            f.seek(0) # Rewind the buffer to the beginning
            onnx_model = onnx.load(f)
            return onnx_model

        except Exception as e:
            print(f"An error occurred during ONNX export: {e}")
            sys.exit(1)

    def verify_onnx_model(self, onnx_model, is_modified=False):
        """
        Verifies an ONNX model (either initial or modified) using ONNX Runtime.
        It uses an in-memory representation of the ONNX model.
        """
        model_type = "modified" if is_modified else "initial"

        try:
            import onnxruntime as ort
            print(f"\nVerifying {model_type} ONNX model with ONNX Runtime...")

            # Save the in-memory model to a temporary buffer for ONNX Runtime to load
            model_buffer = io.BytesIO()
            onnx.save(onnx_model, model_buffer)
            model_buffer.seek(0) # Rewind the buffer

            # Attempt to use ROCmExecutionProvider if available, otherwise fallback
            providers = ['ROCMExecutionProvider', 'CPUExecutionProvider']
            try:
                ort_session = ort.InferenceSession(model_buffer.getvalue(), providers=providers)
                print(f"ONNX model loaded successfully with ONNX Runtime using providers: {ort_session.get_providers()}")
            except Exception as e:
                 print(f"Error loading ONNX model with specified providers: {e}. Attempting with default providers.")
                 ort_session = ort.InferenceSession(model_buffer.getvalue())
                 print(f"ONNX model loaded successfully with ONNX Runtime using default providers: {ort_session.get_providers()}")


            onnx_inputs = ort_session.get_inputs()
            onnx_outputs = ort_session.get_outputs()

            print(f"ONNX Input Name: {onnx_inputs[0].name}")
            print(f"ONNX Input Shape: {onnx_inputs[0].shape}")
            print(f"ONNX Input Type: {onnx_inputs[0].type}")

            print(f"ONNX Output Name: {onnx_outputs[0].name}")
            print(f"ONNX Output Shape: {onnx_outputs[0].shape}")
            print(f"ONNX Output Type: {onnx_outputs[0].type}")

            print(f"\nRunning a dummy inference with ONNX Runtime on {model_type} model...")
            try:
                if is_modified:
                    # Dummy input for modified model (chunky UINT8)
                    dummy_input_np = np.random.randint(0, 256, (1, 576, 752, 4), dtype=np.uint8)
                else:
                    # Dummy input for initial model (planar UINT8)
                    dummy_input_np = np.random.randint(0, 256, (1, 4, 576, 752), dtype=np.uint8)

                ort_inputs = {onnx_inputs[0].name: dummy_input_np}
                ort_outputs = ort_session.run(None, ort_inputs)

                print(f"ONNX Runtime dummy inference successful on {model_type} model.")
                print(f"ONNX Runtime output numpy array shape: {ort_outputs[0].shape}")
                print(f"ONNX Runtime output numpy array dtype: {ort_outputs[0].dtype}")

            except Exception as e:
                print(f"Error during ONNX Runtime dummy inference on {model_type} model: {e}")

        except ImportError:
            print("\nONNX Runtime not found. Install it (`pip install onnxruntime onnxruntime-rocm` for ROCm) to verify the ONNX model.")
        except Exception as e:
            print(f"\nAn unexpected error occurred during ONNX model verification for {model_type} model: {e}")
        finally:
            print(f"\n{model_type.capitalize()} ONNX model verification finished.")


    def modify_onnx_graph_for_chunky(self, model):
        """
        Modifies the ONNX graph (in-memory) to accept chunky (HWC) RGBA input
        and output chunky (HWC) UINT8 RGBA.
        This version is more robust to the PyTorch model's internal preprocessing.
        """
        graph = model.graph
        print("Starting ONNX graph modification for chunky input/output...")

        # --- STEP 1: Modify the INPUT to accept chunky RGBA ([1, H, W, 4]) instead of [1, 4, H, W] ---

        if len(graph.input) != 1:
            print("Error: Model must have exactly one input tensor.")
            sys.exit(1)

        orig_input = graph.input[0]
        orig_input_name = orig_input.name

        orig_shape = []
        for dim in orig_input.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                orig_shape.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                orig_shape.append(dim.dim_param)
            else:
                print("Error: Input shape contains an unspecified dimension.")
                sys.exit(1)

        if len(orig_shape) != 4:
            print(f"Error: Expected input of rank 4, but got rank {len(orig_shape)}")
            sys.exit(1)

        batch_dim, channel_dim, height_dim, width_dim = orig_shape

        new_input_name = "input_rgba_chunky"
        new_input_shape = [batch_dim, height_dim, width_dim, channel_dim]

        orig_input_dtype = orig_input.type.tensor_type.elem_type
        if orig_input_dtype != onnx.TensorProto.DataType.UINT8:
            print(f"Warning: Original input dtype is not UINT8 but "
                  f"{onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[orig_input_dtype].name}")

        new_input_value_info = onnx.helper.make_tensor_value_info(
            new_input_name,
            onnx.TensorProto.DataType.UINT8,
            new_input_shape
        )

        # Create a Transpose node to convert [N, H, W, C] → [N, C, H, W]
        transposed_input_name = f"{new_input_name}_transposed_planar"
        transpose_input_node = onnx.helper.make_node(
            "Transpose",
            inputs=[new_input_name],
            outputs=[transposed_input_name],
            name="Transpose_Chunky_to_Planar",
            perm=[0, 3, 1, 2]
        )

        # --- Dynamic tracing to find the actual first Conv input and intermediate nodes to remove ---
        first_conv_node = None
        for node in graph.node:
            if node.op_type == "Conv":
                first_conv_node = node
                break

        if not first_conv_node:
            print("Error: Could not find the first Conv node in the graph.")
            sys.exit(1)

        target_input_name_for_conv = first_conv_node.input[0]
        nodes_to_remove = []
        visited_nodes = set()
        
        # Helper to find node by output name
        def find_node_by_output(graph_nodes, output_name):
            for n in graph_nodes:
                if output_name in n.output:
                    return n
            return None

        current_node_output = target_input_name_for_conv
        # Trace back from the first Conv's input until we hit the original model input
        # or a constant/initializer
        while True:
            node = find_node_by_output(graph.node, current_node_output)
            if node is None or node.name in visited_nodes: # Stop if node not found or circular reference
                break
            
            # Check if any input of this node is the original input
            if any(inp == orig_input_name for inp in node.input):
                nodes_to_remove.append(node) # This node consumes the original input
                break # We've reached the start of the processing chain

            # Check if this node is a constant or initializer input
            is_constant_input = False
            for input_name in node.input:
                if input_name in [i.name for i in graph.initializer]:
                    is_constant_input = True
                    break
            if is_constant_input:
                break # Stop tracing if we hit a constant

            nodes_to_remove.append(node)
            visited_nodes.add(node.name)

            # Move to the first input of the current node to continue tracing back
            if len(node.input) > 0:
                current_node_output = node.input[0]
            else:
                break # No more inputs to trace


        # Remove the identified intermediate nodes (e.g., Cast, Div, Slice for RGBA/Downsample)
        for node_to_remove in nodes_to_remove:
            if node_to_remove in graph.node: # Ensure it's still in the list before trying to remove
                graph.node.remove(node_to_remove)
            else:
                print(f"Warning: Attempted to remove node {node_to_remove.name} but it was already removed.")

        # --- New Order: Slice RGB, then Slice Downsample, then Cast, then Div ---

        # Add Constant nodes for Slice operator inputs (starts, ends, axes, steps)
        # For rgb_input = x[:, :3, :, :]
        slice_rgb_starts_name = "slice_rgb_starts_constant"
        slice_rgb_starts_node = onnx.helper.make_node(
            "Constant",
            inputs=[], outputs=[slice_rgb_starts_name], name="Constant_SliceRGBStarts",
            value=onnx.helper.make_tensor(name="starts", data_type=onnx.TensorProto.DataType.INT64, dims=[1], vals=[0])
        )
        slice_rgb_ends_name = "slice_rgb_ends_constant"
        slice_rgb_ends_node = onnx.helper.make_node(
            "Constant",
            inputs=[], outputs=[slice_rgb_ends_name], name="Constant_SliceRGBEnds",
            value=onnx.helper.make_tensor(name="ends", data_type=onnx.TensorProto.DataType.INT64, dims=[1], vals=[3])
        )
        slice_rgb_axes_name = "slice_rgb_axes_constant"
        slice_rgb_axes_node = onnx.helper.make_node(
            "Constant",
            inputs=[], outputs=[slice_rgb_axes_name], name="Constant_SliceRGBAxes",
            value=onnx.helper.make_tensor(name="axes", data_type=onnx.TensorProto.DataType.INT64, dims=[1], vals=[1]) # Channel dim
        )
        slice_rgb_steps_name = "slice_rgb_steps_constant"
        slice_rgb_steps_node = onnx.helper.make_node(
            "Constant",
            inputs=[], outputs=[slice_rgb_steps_name], name="Constant_SliceRGBSteps",
            value=onnx.helper.make_tensor(name="steps", data_type=onnx.TensorProto.DataType.INT64, dims=[1], vals=[1])
        )

        # Add Slice node for RGBA to RGB (dropping A channel) - Input is transposed_input_name (uint8)
        sliced_rgb_input_name = "input_rgb_uint8_planar"
        slice_rgb_node = onnx.helper.make_node(
            "Slice",
            inputs=[transposed_input_name, slice_rgb_starts_name, slice_rgb_ends_name, slice_rgb_axes_name, slice_rgb_steps_name],
            outputs=[sliced_rgb_input_name],
            name="Slice_RGBA_to_RGB"
        )
        
        # Add Constant nodes for Slice operator inputs for downsampling (starts, ends, axes, steps)
        # For rgb_input_downsampled = rgb_input[:, :, ::2, ::2]
        slice_downsample_starts_name = "slice_downsample_starts_constant"
        slice_downsample_starts_node = onnx.helper.make_node(
            "Constant",
            inputs=[], outputs=[slice_downsample_starts_name], name="Constant_SliceDownsampleStarts",
            value=onnx.helper.make_tensor(name="starts", data_type=onnx.TensorProto.DataType.INT64, dims=[2], vals=[0, 0])
        )
        slice_downsample_ends_name = "slice_downsample_ends_constant"
        slice_downsample_ends_node = onnx.helper.make_node(
            "Constant",
            inputs=[], outputs=[slice_downsample_ends_name], name="Constant_SliceDownsampleEnds",
            value=onnx.helper.make_tensor(name="ends", data_type=onnx.TensorProto.DataType.INT64, dims=[2], vals=[sys.maxsize, sys.maxsize]) # Use sys.maxsize for "until end"
        )
        slice_downsample_axes_name = "slice_downsample_axes_constant"
        slice_downsample_axes_node = onnx.helper.make_node(
            "Constant",
            inputs=[], outputs=[slice_downsample_axes_name], name="Constant_SliceDownsampleAxes",
            value=onnx.helper.make_tensor(name="axes", data_type=onnx.TensorProto.DataType.INT64, dims=[2], vals=[2, 3]) # Height and Width dims
        )
        slice_downsample_steps_name = "slice_downsample_steps_constant"
        slice_downsample_steps_node = onnx.helper.make_node(
            "Constant",
            inputs=[], outputs=[slice_downsample_steps_name], name="Constant_SliceDownsampleSteps",
            value=onnx.helper.make_tensor(name="steps", data_type=onnx.TensorProto.DataType.INT64, dims=[2], vals=[2, 2])
        )

        # Add Slice node for downsampling - Input is sliced_rgb_input_name (uint8)
        downsampled_rgb_input_name = "input_rgb_uint8_planar_downsampled"
        slice_downsample_node = onnx.helper.make_node(
            "Slice",
            inputs=[sliced_rgb_input_name, slice_downsample_starts_name, slice_downsample_ends_name, slice_downsample_axes_name, slice_downsample_steps_name],
            outputs=[downsampled_rgb_input_name],
            name="Slice_Downsample_RGB"
        )

        # Create new Cast(to FP16) and Div nodes for the optimized input path
        # Input to Cast is now downsampled_rgb_input_name (uint8)
        cast_to_fp16_input_name = "input_rgb_float16_planar_downsampled"
        cast_to_fp16_node = onnx.helper.make_node(
            "Cast",
            inputs=[downsampled_rgb_input_name], # Corrected input here!
            outputs=[cast_to_fp16_input_name],
            name="Cast_Input_To_FP16",
            to=onnx.TensorProto.DataType.FLOAT16
        )

        div_by_255_constant_name = "div_by_255_constant"
        div_by_255_constant_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=[div_by_255_constant_name],
            name="Constant_Div255",
            value=onnx.helper.make_tensor(
                name=div_by_255_constant_name,
                data_type=onnx.TensorProto.DataType.FLOAT16,
                dims=[],
                vals=[np.float16(255.0).item()]
            )
        )

        normalized_rgb_input_name = "input_rgb_float16_normalized_downsampled" # Changed name to reflect RGB
        div_node = onnx.helper.make_node(
            "Div",
            inputs=[cast_to_fp16_input_name, div_by_255_constant_name],
            outputs=[normalized_rgb_input_name],
            name="Div_Input_By_255"
        )

        # Update the input of the first Conv node to receive the downsampled RGB input
        for i, input_name in enumerate(first_conv_node.input):
            if input_name == target_input_name_for_conv: # target_input_name_for_conv is the original input to the first Conv
                first_conv_node.input[i] = normalized_rgb_input_name # Corrected input here!
                break

        graph.input.remove(orig_input)
        graph.input.extend([new_input_value_info])

        # Insert the new nodes at the very front of graph.node list in correct order:
        # Transpose -> RGB Slice Constants -> RGB Slice -> Downsample Slice Constants -> Downsample Slice -> Constant for Div -> Cast -> Div
        insert_idx = 0
        graph.node.insert(insert_idx, transpose_input_node) ; insert_idx += 1
        graph.node.insert(insert_idx, slice_rgb_starts_node) ; insert_idx += 1
        graph.node.insert(insert_idx, slice_rgb_ends_node) ; insert_idx += 1
        graph.node.insert(insert_idx, slice_rgb_axes_node) ; insert_idx += 1
        graph.node.insert(insert_idx, slice_rgb_steps_node) ; insert_idx += 1
        graph.node.insert(insert_idx, slice_rgb_node) ; insert_idx += 1
        graph.node.insert(insert_idx, slice_downsample_starts_node) ; insert_idx += 1
        graph.node.insert(insert_idx, slice_downsample_ends_node) ; insert_idx += 1
        graph.node.insert(insert_idx, slice_downsample_axes_node) ; insert_idx += 1
        graph.node.insert(insert_idx, slice_downsample_steps_node) ; insert_idx += 1
        graph.node.insert(insert_idx, slice_downsample_node) ; insert_idx += 1
        graph.node.insert(insert_idx, div_by_255_constant_node) ; insert_idx += 1 # Constant must be before its use
        graph.node.insert(insert_idx, cast_to_fp16_node) ; insert_idx += 1
        graph.node.insert(insert_idx, div_node) ; insert_idx += 1


        print(f"Replaced model input '{orig_input_name}' with chunky input '{new_input_name}' and optimized input preprocessing including RGBA to RGB slice and downsampling.")

        # Add ValueInfo for the intermediate transposed and normalized tensors
        graph.value_info.append(onnx.helper.make_tensor_value_info(
            transposed_input_name,
            onnx.TensorProto.DataType.UINT8, # Still uint8 after transpose
            [batch_dim, channel_dim, height_dim, width_dim]
        ))
        graph.value_info.append(onnx.helper.make_tensor_value_info(
            sliced_rgb_input_name, # This is now 3 channels after RGB Slice, still uint8
            onnx.TensorProto.DataType.UINT8,
            [batch_dim, 3, height_dim, width_dim]
        ))
        graph.value_info.append(onnx.helper.make_tensor_value_info(
            downsampled_rgb_input_name, # This is now 3 channels at half resolution, still uint8
            onnx.TensorProto.DataType.UINT8,
            [batch_dim, 3, height_dim // 2, width_dim // 2]
        ))
        graph.value_info.append(onnx.helper.make_tensor_value_info(
            cast_to_fp16_input_name, # This is after cast, now float16
            onnx.TensorProto.DataType.FLOAT16,
            [batch_dim, 3, height_dim // 2, width_dim // 2]
        ))
        graph.value_info.append(onnx.helper.make_tensor_value_info(
            normalized_rgb_input_name, # This is after div, still float16
            onnx.TensorProto.DataType.FLOAT16,
            [batch_dim, 3, height_dim // 2, width_dim // 2]
        ))


        # --- STEP 2: Modify the OUTPUT to convert [1, 4, H, W] FLOAT16 → UINT8 → chunky [1, H, W, 4] ---
        # Note: Original output is assumed to be 4 channels (RGBA) from the model's output.
        # If your model outputs 3 channels, you would need to adjust the output processing as well.

        if len(graph.output) != 1:
            print("Error: Model must have exactly one output tensor before modification.")
            sys.exit(1)

        orig_output = graph.output[0]
        orig_output_name = orig_output.name

        orig_out_shape = []
        for dim in orig_output.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                orig_out_shape.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                orig_out_shape.append(dim.dim_param)
            else:
                print("Error: Output shape contains an unspecified dimension.")
                sys.exit(1)

        orig_out_dtype = orig_output.type.tensor_type.elem_type
        # This will be Float16 due to model.half() and the output scaling.
        if orig_out_dtype != onnx.TensorProto.DataType.FLOAT16:
            print(f"Warning: Original output dtype is not FLOAT16 but "
                  f"{onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[orig_out_dtype].name}")

        graph.output.remove(orig_output)

        clip_min_name = "clip_min_constant"
        clip_max_name = "clip_max_constant"
        clip_min_val = np.float16(0.0)
        clip_max_val = np.float16(255.0)

        clip_min_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=[clip_min_name],
            name="Constant_ClipMin",
            value=onnx.helper.make_tensor(
                name=clip_min_name,
                data_type=onnx.TensorProto.DataType.FLOAT16,
                dims=[],
                vals=[clip_min_val.item()]
            )
        )
        clip_max_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=[clip_max_name],
            name="Constant_ClipMax",
            value=onnx.helper.make_tensor(
                name=clip_max_name,
                data_type=onnx.TensorProto.DataType.FLOAT16,
                dims=[],
                vals=[clip_max_val.item()]
            )
        )

        clipped_output_name = "output_rgba_float16_clipped"
        clip_node = onnx.helper.make_node(
            "Clip",
            inputs=[orig_output_name, clip_min_name, clip_max_name],
            outputs=[clipped_output_name],
            name="Clip_Output"
        )

        cast_uint8_output_name = "output_rgba_uint8_planar"
        cast_node = onnx.helper.make_node(
            "Cast",
            inputs=[clipped_output_name],
            outputs=[cast_uint8_output_name],
            name="Cast_To_Uint8",
            to=onnx.TensorProto.DataType.UINT8
        )

        transposed_output_name = "output_rgba_uint8_chunky"
        transpose_output_node = onnx.helper.make_node(
            "Transpose",
            inputs=[cast_uint8_output_name],
            outputs=[transposed_output_name],
            name="Transpose_Planar_to_Chunky",
            perm=[0, 2, 3, 1]
        )

        graph.node.extend([
            clip_min_node,
            clip_max_node,
            clip_node,
            cast_node,
            transpose_output_node
        ])

        if len(orig_out_shape) != 4:
            print(f"Error: Expected original output rank 4, but got {len(orig_out_shape)}")
            sys.exit(1)

        batch_o, channel_o, height_o, width_o = orig_out_shape
        new_out_shape = [batch_o, height_o, width_o, channel_o]

        new_output_value_info = onnx.helper.make_tensor_value_info(
            transposed_output_name,
            onnx.TensorProto.DataType.UINT8,
            new_out_shape
        )
        graph.output.extend([new_output_value_info])

        print(f"Added Clip, Cast, and Transpose to convert '{orig_output_name}' → '{transposed_output_name}' and set as new output.")

        # Return the modified model object
        return model

    def save_onnx_model(self, model):
        """
        Validates and saves the modified ONNX model to the specified path.
        """
        print("\nValidating the ONNX model before saving...")
        try:
            onnx.checker.check_model(model)
            print("ONNX model is valid.")
        except Exception as e:
            print(f"Error validating ONNX model: {e}")
            print("Saving model despite validation errors for inspection.") # Allow saving even with errors

        print(f"\nSaving ONNX model to {self.output_onnx_path}...")
        try:
            onnx.save(model, os.path.abspath(self.output_onnx_path))
            print("ONNX model saved successfully.")
        except Exception as e:
            print(f"Error saving ONNX model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX and modify for chunky RGBA input/output.")
    parser.add_argument(
        "--pytorch_path",
        type=str,
        required=True,
        help="Path to the trained PyTorch model file (e.g., basic_conv6/best_conv6.pt)"
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        default="conv6_chunky.onnx", # This will be the final output path
        help="Path to save the final modified ONNX model with chunky input/output"
    )

    args = parser.parse_args()

    converter = ONNXConverter(
        pytorch_model_path=args.pytorch_path,
        output_onnx_path=args.onnx_path
    )

    print("Step 1: Load PyTorch model and export to initial ONNX in memory")
    converter.load_pytorch_model()
    time.sleep(1) # Reduced sleep for quicker execution
    intermediate_onnx_model = converter.export_to_onnx_in_memory()
    time.sleep(1) # Reduced sleep for quicker execution

    print("Step 2: Modify the ONNX graph for chunky input/output (in memory)")
    modified_onnx_model = converter.modify_onnx_graph_for_chunky(intermediate_onnx_model)
    time.sleep(1) # Reduced sleep for quicker execution

    print("Step 3: Simplify the ONNX model (including constant folding)")
    print("\nAttempting to simplify the ONNX model (including constant folding)...")
    try:
        import onnxsim
        print("onnx-simplifier found. Proceeding with simplification.")
        # onnxsim.simplify returns (simplified_model, check_ok)
        # It's important to pass graph.node as a list or tuple for iteration, not modify during iteration.
        simplified_model, check = onnxsim.simplify(modified_onnx_model)

        if check:
            print("ONNX model simplified successfully!")
            final_onnx_model = simplified_model
        else:
            print("Warning: ONNX model simplification failed. Using the unsimplified modified model.")
            final_onnx_model = modified_onnx_model

    except ImportError:
        print("Warning: onnx-simplifier not found. Please install it (`pip install onnx-simplifier`) to enable graph simplification.")
        print("Skipping simplification. Using the unsimplified modified model.")
        final_onnx_model = modified_onnx_model
    except Exception as e:
        print(f"Error during ONNX model simplification: {e}. Using the unsimplified modified model.")
        final_onnx_model = modified_onnx_model

    print("Step 4: Verify and save the FINAL (potentially simplified) ONNX model")
    converter.verify_onnx_model(final_onnx_model, is_modified=True)
    time.sleep(1) # Reduced sleep for quicker execution
    converter.save_onnx_model(final_onnx_model)

    print("\nFull ONNX conversion, modification, and potential simplification process completed.")