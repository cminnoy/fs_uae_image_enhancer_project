import torch
import torch.nn as nn
import torch.onnx
import sys
import onnx
import onnx.helper
import onnx.mapping
import numpy as np
import argparse
import io # For saving ONNX model to a byte stream in memory for verification

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
            self.model = torch.load(self.pytorch_model_path, weights_only=False)
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

        if self.device.type == 'cuda':
            self.model.half()
            print("Model parameters converted to Half precision (FP16) for export.")
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
                input_dtypes=[torch.uint8],
                output_dtypes=[torch.float16]
            )
            print("Model exported successfully to FP16 ONNX format in memory!")

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

        # Find the first Conv node and its actual input
        first_conv_node = None
        target_input_name_for_conv = None # This will store the name of the tensor that actually feeds the first Conv
        nodes_to_remove = []

        # Identify the node that directly feeds the first Conv
        for node in graph.node:
            # Assuming the first Conv node's input is a tensor derived from a Cast node output
            # This logic needs to be robust. Using startswith to catch variations like /Cast_1_output_0 or /Cast_output_0
            if node.op_type == "Conv" and node.input[0].startswith("/Cast"): # Broader check for Cast node output
                first_conv_node = node
                target_input_name_for_conv = node.input[0]
                break

        if not first_conv_node or not target_input_name_for_conv:
            print("Error: Could not find the first Conv node or its expected input tensor.")
            sys.exit(1)

        # Trace back from target_input_name_for_conv to collect nodes to remove
        current_output_to_trace = target_input_name_for_conv
        
        # Helper to find node by output name
        def find_node_by_output(graph_nodes, output_name):
            for n in graph_nodes:
                if output_name in n.output:
                    return n
            return None

        # Trace back to collect nodes to remove
        while True:
            node_to_add_to_remove = find_node_by_output(graph.node, current_output_to_trace)
            # Stop if we reach a node that directly consumes the original input or a constant
            if node_to_add_to_remove is None or node_to_add_to_remove.output[0] == orig_input_name:
                break
            nodes_to_remove.append(node_to_add_to_remove)
            
            # Find the next input to trace back
            # Ensure we don't try to trace back through constant inputs or if no more inputs
            if len(node_to_add_to_remove.input) > 0 and node_to_add_to_remove.input[0] != orig_input_name:
                current_output_to_trace = node_to_add_to_remove.input[0]
            else:
                break # Reached a constant or original input

        # Remove identified nodes in reverse order to avoid issues
        for node_to_remove in reversed(nodes_to_remove):
            graph.node.remove(node_to_remove)

        # Create new Cast(to FP16) and Div nodes for the optimized input path
        cast_to_fp16_input_name = "input_rgba_float16_planar"
        cast_to_fp16_node = onnx.helper.make_node(
            "Cast",
            inputs=[transposed_input_name],
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

        normalized_rgba_input_name = "input_rgba_float16_normalized"
        div_node = onnx.helper.make_node(
            "Div",
            inputs=[cast_to_fp16_input_name, div_by_255_constant_name],
            outputs=[normalized_rgba_input_name],
            name="Div_Input_By_255"
        )
        
        # --- NEW: Add Constant nodes for Slice operator inputs (starts, ends, axes) ---
        slice_starts_constant_name = "slice_starts_constant"
        slice_starts_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=[slice_starts_constant_name],
            name="Constant_SliceStarts",
            value=onnx.helper.make_tensor(
                name="starts",
                data_type=onnx.TensorProto.DataType.INT64,
                dims=[1],
                vals=[0]
            )
        )

        slice_ends_constant_name = "slice_ends_constant"
        slice_ends_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=[slice_ends_constant_name],
            name="Constant_SliceEnds",
            value=onnx.helper.make_tensor(
                name="ends",
                data_type=onnx.TensorProto.DataType.INT64,
                dims=[1],
                vals=[3]
            )
        )

        slice_axes_constant_name = "slice_axes_constant"
        slice_axes_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=[slice_axes_constant_name],
            name="Constant_SliceAxes",
            value=onnx.helper.make_tensor(
                name="axes",
                data_type=onnx.TensorProto.DataType.INT64,
                dims=[1],
                vals=[1] # Channel dimension in NCHW
            )
        )

        # --- MODIFIED SLICE NODE DEFINITION to use constant inputs ---
        sliced_rgb_input_name = "input_rgb_float16_normalized"
        slice_rgb_node = onnx.helper.make_node(
            "Slice",
            inputs=[normalized_rgba_input_name, 
                    slice_starts_constant_name, 
                    slice_ends_constant_name, 
                    slice_axes_constant_name],
            outputs=[sliced_rgb_input_name],
            name="Slice_RGBA_to_RGB"
        )


        # Update the input of the first Conv node to receive the sliced RGB input
        for i, input_name in enumerate(first_conv_node.input):
            if input_name == target_input_name_for_conv:
                first_conv_node.input[i] = sliced_rgb_input_name
                break

        graph.input.remove(orig_input)
        graph.input.extend([new_input_value_info])

        # Insert the new nodes at the very front of graph.node list in order:
        # Transpose -> Constant for Div -> Cast -> Div -> Constant_SliceStarts -> Constant_SliceEnds -> Constant_SliceAxes -> Slice
        graph.node.insert(0, transpose_input_node)
        graph.node.insert(1, div_by_255_constant_node)
        graph.node.insert(2, cast_to_fp16_node)
        graph.node.insert(3, div_node)
        # Insert the new constant nodes for slice *before* the slice node itself
        graph.node.insert(4, slice_starts_node)
        graph.node.insert(5, slice_ends_node)
        graph.node.insert(6, slice_axes_node)
        graph.node.insert(7, slice_rgb_node) # Insert the Slice node here


        # Add ValueInfo for the intermediate transposed and normalized tensors
        graph.value_info.append(onnx.helper.make_tensor_value_info(
            transposed_input_name,
            onnx.TensorProto.DataType.UINT8, # Still uint8 after transpose
            [batch_dim, channel_dim, height_dim, width_dim]
        ))
        graph.value_info.append(onnx.helper.make_tensor_value_info(
            cast_to_fp16_input_name,
            onnx.TensorProto.DataType.FLOAT16,
            [batch_dim, channel_dim, height_dim, width_dim]
        ))
        graph.value_info.append(onnx.helper.make_tensor_value_info(
            normalized_rgba_input_name, # This is still 4 channels after Div
            onnx.TensorProto.DataType.FLOAT16,
            [batch_dim, channel_dim, height_dim, width_dim]
        ))
        graph.value_info.append(onnx.helper.make_tensor_value_info(
            sliced_rgb_input_name, # This is now 3 channels after Slice
            onnx.TensorProto.DataType.FLOAT16,
            [batch_dim, 3, height_dim, width_dim] # Sliced to 3 channels
        ))


        print(f"Replaced model input '{orig_input_name}' with chunky input '{new_input_name}' and optimized input preprocessing including RGBA to RGB slice.")

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
            onnx.save(model, self.output_onnx_path)
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

    # Step 1: Load PyTorch model and export to initial ONNX in memory
    converter.load_pytorch_model()
    intermediate_onnx_model = converter.export_to_onnx_in_memory()

    # Step 2: Modify the ONNX graph for chunky input/output (in memory)
    modified_onnx_model = converter.modify_onnx_graph_for_chunky(intermediate_onnx_model)

    # Step 3: Simplify the ONNX model (including constant folding)
    print("\nAttempting to simplify the ONNX model (including constant folding)...")
    try:
        import onnxsim
        print("onnx-simplifier found. Proceeding with simplification.")
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

    # Step 4: Verify and save the FINAL (potentially simplified) ONNX model
    converter.verify_onnx_model(final_onnx_model, is_modified=True)
    converter.save_onnx_model(final_onnx_model)

    print("\nFull ONNX conversion, modification, and potential simplification process completed.")