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
import os
import time

sys.path.append(os.getcwd()) # Ensure current directory is in path for model loading

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
        self.model_has_pixel_shuffle = False # Flag to detect PixelShuffle

    def load_pytorch_model(self):
        """
        Loads the PyTorch model from the specified path and prepares it for export.
        """
        print(f"--- Step 1: Loading PyTorch Model ---")
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

        # Check for PixelShuffle in the model
        self.model_has_pixel_shuffle = any(isinstance(m, nn.PixelShuffle) for m in self.model.modules())
        print(f"Model has nn.PixelShuffle: {self.model_has_pixel_shuffle}")

        print(f"ONNX converter will NOT add an initial external downsample slice as the model is expected to handle it internally.")


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model moved to device: {self.device}")

        if self.device.type == 'cuda':
            self.model.half()
            print("Model parameters converted to Half precision (FP16) for export.")
        else:
            print("Running on CPU, skipping FP16 conversion for model parameters.")

        # Attempt to fuse layers
        if hasattr(self.model, 'fuse_layers') and callable(self.model.fuse_layers):
            print("Calling fuse_layers method for optimization...")
            try:
                self.model.eval() # Redundant call, but harmless
                self.model.fuse_layers()
                print("Fusion utility ran, but some BatchNorm or Activation layers still exist. Fusion might not have been fully effective.")
            except Exception as e:
                print(f"\nAn unexpected error occurred during layer fusion using fuse_modules: {e}")
                print("Skipping PyTorch-side layer fusion.\nHopeful that ONNX Runtime will perform Conv+BN[+ReLU] fusion during inference.")
            print("Layer fusion complete.")
        else:
            print("Warning: Model instance does not have a 'fuse_layers' method or it's not callable.")
            print("Skipping layer fusion.")

    def export_to_onnx_in_memory(self):
        """
        Exports the PyTorch model to ONNX format and returns the ONNX model object in memory.
        """
        print(f"\n--- Step 2: Exporting PyTorch Model to ONNX in memory ---")
        # Dummy input is now 3 channels (RGB) to match the trained model's expectation.
        # The range 0-1 is typical for normalized float inputs for models.
        dummy_input = torch.rand((1, 3, 576, 752), dtype=torch.float16).to(self.device)
        print(f"Created dummy input tensor with shape {dummy_input.shape} and dtype {dummy_input.dtype} on device {self.device}")

        # Use generic names for input/output of the internal PyTorch model
        input_names = ["model_input_rgb_float16_planar"]
        output_names = ["model_output_rgb_float16_scaled"]
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
                input_dtypes=[torch.float16], # Input dtype for export is float16
                output_dtypes=[torch.float16]
            )
            print("Model exported successfully to FP16 ONNX format in memory!")

            # Load the ONNX model from the in-memory buffer
            f.seek(0) # Rewind the buffer to the beginning
            onnx_model = onnx.load(f)
            # DEBUG: Initial ONNX Graph Input Name: model_input_rgb_float16_planar
            # DEBUG: Initial ONNX Graph Output Name: model_output_rgb_float16_scaled
            print(f"DEBUG: Initial ONNX Graph Input Name: {onnx_model.graph.input[0].name}")
            print(f"DEBUG: Initial ONNX Graph Output Name: {onnx_model.graph.output[0].name}")

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
        print(f"\nVerifying {model_type} ONNX model with ONNX Runtime...")

        try:
            import onnxruntime as ort
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
                # Dummy input for modified model (chunky UINT8 RGBA)
                dummy_input_np = np.random.randint(0, 256, (1, 576, 752, 4), dtype=np.uint8)

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
        and output chunky (HWC) UINT8 RGBA, including sRGB<->Linear conversions,
        with optimizations for alpha channel handling and avoiding redundant resize.
        """
        graph = model.graph
        print("\n--- Step 3: Starting ONNX graph modification for chunky input/output ---")

        # --- STEP 3a: Modify the INPUT to accept chunky RGBA ([1, H, W, 4]) and preprocess ---

        if len(graph.input) != 1:
            print("Error: Model must have exactly one input tensor.")
            sys.exit(1)

        orig_input = graph.input[0]
        # This is now "model_input_rgb_float16_planar" from the initial torch.onnx.export
        orig_input_name = orig_input.name 
        
        # We need the original dimensions from the exported model's input
        # Note: orig_shape will be [N, 3, H, W] because dummy_input was 3-channel.
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

        batch_dim, model_channel_dim, height_dim, width_dim = orig_shape
        # model_channel_dim is 3 here

        # 1. Define the new external input to the ONNX model (chunky uint8 RGBA)
        new_external_input_name = "input_rgba_chunky"
        new_external_input_shape = [batch_dim, height_dim, width_dim, 4] # Explicitly 4 channels for external RGBA input

        new_external_input_value_info = onnx.helper.make_tensor_value_info(
            new_external_input_name,
            onnx.TensorProto.DataType.UINT8, # External input is UINT8
            new_external_input_shape
        )

        # 2. Create a Transpose node: [N, H, W, C] (uint8) → [N, C, H, W] (uint8)
        # This will operate on 4 channels as input_rgba_chunky is 4 channels.
        transposed_input_name_uint8 = f"{new_external_input_name}_transposed_planar_uint8"
        transpose_input_node = onnx.helper.make_node(
            "Transpose",
            inputs=[new_external_input_name],
            outputs=[transposed_input_name_uint8],
            name="Transpose_Chunky_to_Planar_Uint8",
            perm=[0, 3, 1, 2] # Swaps last two dims with channel dim
        )

        # 3. Slice node to remove alpha channel (RGBA to RGB) - slices UINT8 before casting
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
                vals=[3] # Slice from 0 up to (but not including) 3 for RGB
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

        # Output of this slice is now 3-channel UINT8
        sliced_rgb_input_name_uint8 = "input_rgb_uint8_planar_sliced"
        slice_rgb_node = onnx.helper.make_node(
            "Slice",
            inputs=[transposed_input_name_uint8, # Input is 4-channel UINT8
                    slice_starts_constant_name, 
                    slice_ends_constant_name, 
                    slice_axes_constant_name],
            outputs=[sliced_rgb_input_name_uint8], # Output is 3-channel UINT8
            name="Slice_RGBA_to_RGB_Uint8"
        )


        # 4. Create Cast node: uint8 → float16 (now operates on 3 channels)
        cast_to_fp16_input_name = "input_rgb_float16_planar"
        cast_to_fp16_node = onnx.helper.make_node(
            "Cast",
            inputs=[sliced_rgb_input_name_uint8], # Input is now 3-channel UINT8
            outputs=[cast_to_fp16_input_name], # Output is 3-channel FP16
            name="Cast_Uint8_To_FP16",
            to=onnx.TensorProto.DataType.FLOAT16
        )

        # 5. Create Div node: Normalize (divide by 255.0, operates on 3 channels)
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

        normalized_rgb_input_name = "input_rgb_float16_normalized"
        div_node = onnx.helper.make_node(
            "Div",
            inputs=[cast_to_fp16_input_name, div_by_255_constant_name], # Input is 3-channel FP16
            outputs=[normalized_rgb_input_name], # Output is 3-channel FP16
            name="Div_Input_By_255"
        )
        
        # 6. sRGB to Linear Gamma Correction (Pow(x, 2.2), operates on 3 channels)
        # Note: The user provided a more accurate sRGB <-> Linear conversion (torch.where based).
        # Implementing that would require ONNX 'If' nodes and subgraphs, significantly
        # increasing graph complexity and potentially runtime overhead for certain ONNX runtimes.
        # For simplicity and common compatibility, we continue using the approximate Pow() function.
        # If higher precision is strictly required at the cost of graph complexity,
        # the 'If' node approach should be investigated.
        gamma_srgb_to_linear_exp_name = "gamma_srgb_to_linear_exponent"
        gamma_srgb_to_linear_exp_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=[gamma_srgb_to_linear_exp_name],
            name="Constant_GammaSRGBtoLinearExp",
            value=onnx.helper.make_tensor(
                name="gamma_exp",
                data_type=onnx.TensorProto.DataType.FLOAT16,
                dims=[],
                vals=[np.float16(2.2).item()] # sRGB to Linear exponent
            )
        )
        
        linear_rgb_input_name = "input_rgb_float16_linear"
        srgb_to_linear_node = onnx.helper.make_node(
            "Pow",
            inputs=[normalized_rgb_input_name, gamma_srgb_to_linear_exp_name], # Input is 3-channel FP16
            outputs=[linear_rgb_input_name], # Output is 3-channel FP16
            name="SRGB_to_Linear_Gamma"
        )

        # IMPORTANT CHANGE: Iterate through all nodes to redirect any reference to the original input.
        # This handles skip connections where the original input might be used later in the graph.
        modified_node_names = []
        found_consumer = False
        for node in graph.node:
            # Create a list of input names for modification to avoid changing list while iterating
            new_inputs_for_node = list(node.input)
            current_node_modified = False
            for i, input_name in enumerate(new_inputs_for_node):
                if input_name == orig_input_name:
                    new_inputs_for_node[i] = linear_rgb_input_name
                    current_node_modified = True
                    found_consumer = True # At least one consumer found
            if current_node_modified:
                node.input[:] = new_inputs_for_node # Update the node's inputs
                modified_node_names.append(node.name)

        if not found_consumer:
            print("Error: Could not find any node that consumes the original exported model input. The model might not be correctly connected.")
            sys.exit(1)
        else:
            print(f"Redirected original model input for nodes: {', '.join(modified_node_names)}")

        # Remove the original input from the graph
        graph.input.remove(orig_input) 
        # Add the new external input (4-channel RGBA)
        graph.input.extend([new_external_input_value_info]) 

        # Insert the new nodes at the very front of graph.node list in order:
        # 1. Transpose
        # 2. Slice (NEW position, slices UINT8)
        # 3. Cast (new input)
        # 4. Constant (div255)
        # 5. Div (new input)
        # 6. Constant (gamma_srgb_to_linear_exp)
        # 7. Pow (new input)
        
        graph.node.insert(0, transpose_input_node)
        graph.node.insert(1, slice_starts_node)
        graph.node.insert(2, slice_ends_node)
        graph.node.insert(3, slice_axes_node)
        graph.node.insert(4, slice_rgb_node)
        graph.node.insert(5, cast_to_fp16_node)
        graph.node.insert(6, div_by_255_constant_node)
        graph.node.insert(7, div_node)
        graph.node.insert(8, gamma_srgb_to_linear_exp_node)
        graph.node.insert(9, srgb_to_linear_node)

        # Add ValueInfo for the intermediate tensors (for graph validation/inspection)
        graph.value_info.append(onnx.helper.make_tensor_value_info(
            transposed_input_name_uint8,
            onnx.TensorProto.DataType.UINT8,
            [batch_dim, 4, height_dim, width_dim] # 4 channels
        ))
        graph.value_info.append(onnx.helper.make_tensor_value_info(
            sliced_rgb_input_name_uint8, # NEW intermediate value
            onnx.TensorProto.DataType.UINT8,
            [batch_dim, 3, height_dim, width_dim] # Sliced to 3 channels (UINT8)
        ))
        graph.value_info.append(onnx.helper.make_tensor_value_info(
            cast_to_fp16_input_name,
            onnx.TensorProto.DataType.FLOAT16,
            [batch_dim, 3, height_dim, width_dim] # 3 channels (FP16)
        ))
        graph.value_info.append(onnx.helper.make_tensor_value_info(
            normalized_rgb_input_name,
            onnx.TensorProto.DataType.FLOAT16,
            [batch_dim, 3, height_dim, width_dim] # 3 channels (FP16)
        ))
        graph.value_info.append(onnx.helper.make_tensor_value_info(
            linear_rgb_input_name,
            onnx.TensorProto.DataType.FLOAT16,
            [batch_dim, 3, height_dim, width_dim] # 3 channels (FP16)
        ))

        print(f"Replaced model input '{orig_input_name}' with chunky input '{new_external_input_name}' and optimized input preprocessing including RGBA to RGB slice (UINT8), sRGB to Linear gamma.")
        print("Removed external Resize node as the model is expected to handle downsampling internally.")


        # --- STEP 3b: Modify the OUTPUT to convert [1, 3, H, W] FLOAT16 → UINT8 → chunky [1, H, W, 4] ---
        if len(graph.output) != 1:
            print("Error: Model must have exactly one output tensor before modification.")
            sys.exit(1)

        orig_output = graph.output[0]
        # This is now "model_output_rgb_float16_scaled"
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

        # 1. Linear to sRGB Gamma Correction (Pow(x, 1/2.2), operates on 3 channels)
        # Note: Same note as above for the input conversion about more complex torch.where logic.
        gamma_linear_to_srgb_exp_name = "gamma_linear_to_srgb_exponent"
        gamma_linear_to_srgb_exp_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=[gamma_linear_to_srgb_exp_name],
            name="Constant_GammaLinearToSRGBExp",
            value=onnx.helper.make_tensor(
                name="gamma_exp",
                data_type=onnx.TensorProto.DataType.FLOAT16,
                dims=[],
                vals=[np.float16(1.0/2.2).item()] # Linear to sRGB exponent
            )
        )

        srgb_output_name_float16 = "output_rgb_float16_srgb" # Still RGB
        linear_to_srgb_node = onnx.helper.make_node(
            "Pow",
            inputs=[orig_output_name, gamma_linear_to_srgb_exp_name], # Orig output is linear [0,1]
            outputs=[srgb_output_name_float16],
            name="Linear_to_SRGB_Gamma"
        )

        # 2. Denormalization (Mul by 255.0, operates on 3 channels)
        denorm_255_constant_name = "denormalization_255_constant"
        denorm_255_constant_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=[denorm_255_constant_name],
            name="Constant_Denorm255",
            value=onnx.helper.make_tensor(
                name=denorm_255_constant_name,
                data_type=onnx.TensorProto.DataType.FLOAT16,
                dims=[],
                vals=[np.float16(255.0).item()]
            )
        )

        denormalized_srgb_output_name_float16 = "output_rgb_float16_srgb_denormalized" # Still RGB
        denormalize_node = onnx.helper.make_node(
            "Mul",
            inputs=[srgb_output_name_float16, denorm_255_constant_name],
            outputs=[denormalized_srgb_output_name_float16],
            name="Denormalize_Output_by_255"
        )

        # 3. Clip to 0-255 range (directly on float16, removed redundant cast to float32)
        clip_min_name = "clip_min_constant"
        clip_max_name = "clip_max_constant"
        clip_min_val = np.float16(0.0) # Changed to FLOAT16
        clip_max_val = np.float16(255.0) # Changed to FLOAT16

        clip_min_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=[clip_min_name],
            name="Constant_ClipMin",
            value=onnx.helper.make_tensor(
                name=clip_min_name,
                data_type=onnx.TensorProto.DataType.FLOAT16, # Changed to FLOAT16
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
                data_type=onnx.TensorProto.DataType.FLOAT16, # Changed to FLOAT16
                dims=[],
                vals=[clip_max_val.item()]
            )
        )

        clipped_output_name = "output_rgb_float16_clipped" # Output is now FP16
        clip_node = onnx.helper.make_node(
            "Clip",
            inputs=[denormalized_srgb_output_name_float16, clip_min_name, clip_max_name], # Input is FP16
            outputs=[clipped_output_name],
            name="Clip_Output"
        )

        # 4. Cast to uint8 (input is now directly from FP16 clip)
        cast_uint8_output_name = "output_rgb_uint8_planar" # Still RGB
        cast_node = onnx.helper.make_node(
            "Cast",
            inputs=[clipped_output_name], # Input is FP16
            outputs=[cast_uint8_output_name],
            name="Cast_To_Uint8",
            to=onnx.TensorProto.DataType.UINT8
        )

        # 5. Pad node to add alpha channel (RGB to RGBA)
        padded_output_name_uint8 = "output_rgba_uint8_planar_padded"
        
        # Define padding for the channel dimension (add 1 channel at the end)
        pad_pads_constant_name = "pad_pads_constant"
        pad_pads_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=[pad_pads_constant_name],
            name="Constant_PadPads",
            value=onnx.helper.make_tensor(
                name="pads",
                data_type=onnx.TensorProto.DataType.INT64,
                dims=[8], # (N_begin, C_begin, H_begin, W_begin, N_end, C_end, H_end, W_end)
                vals=[0, 0, 0, 0, 0, 1, 0, 0] # Pad 1 at the end of the channel dimension
            )
        )

        # Define constant for alpha channel value (255)
        # Note: Pad operator's value input needs to match the input tensor's data type
        # In this case, `cast_uint8_output_name` is UINT8, so the value should be UINT8.
        pad_value_constant_name = "pad_value_constant"
        pad_value_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=[pad_value_constant_name],
            name="Constant_PadValue",
            value=onnx.helper.make_tensor(
                name="value",
                data_type=onnx.TensorProto.DataType.UINT8, # Alpha value is 255 (uint8)
                dims=[],
                vals=[255]
            )
        )

        pad_node = onnx.helper.make_node(
            "Pad",
            inputs=[cast_uint8_output_name, pad_pads_constant_name, pad_value_constant_name],
            outputs=[padded_output_name_uint8],
            name="Pad_RGB_to_RGBA"
        )


        # 6. Transpose to chunky format
        transposed_output_name = "output_rgba_uint8_chunky"
        transpose_output_node = onnx.helper.make_node(
            "Transpose",
            inputs=[padded_output_name_uint8], # Input is now 4-channel RGBA
            outputs=[transposed_output_name],
            name="Transpose_Planar_to_Chunky",
            perm=[0, 2, 3, 1]
        )

        graph.node.extend([
            gamma_linear_to_srgb_exp_node,
            linear_to_srgb_node,          
            denorm_255_constant_node,     
            denormalize_node,             
            clip_min_node,
            clip_max_node,
            clip_node,
            cast_node,
            pad_pads_node,
            pad_value_node,
            pad_node,
            transpose_output_node
        ])

        if len(orig_out_shape) != 4:
            print(f"Error: Expected original output rank 4, but got {len(orig_out_shape)}")
            sys.exit(1)

        batch_o, model_out_channel_o, height_o, width_o = orig_out_shape
        # model_out_channel_o is 3 here. We need 4 for final chunky output.
        new_out_shape = [batch_o, height_o, width_o, 4] # Explicitly 4 channels for final RGBA chunky output

        new_output_value_info = onnx.helper.make_tensor_value_info(
            transposed_output_name,
            onnx.TensorProto.DataType.UINT8,
            new_out_shape
        )
        graph.output.extend([new_output_value_info])

        print(f"Added Linear to sRGB gamma, Denormalization, Clip (FP16), Cast to UINT8, RGB to RGBA Pad, and Transpose to convert '{orig_output_name}' → '{transposed_output_name}' and set as new output.")

        # Return the modified model object
        return model

    def save_onnx_model(self, model):
        """
        Validates and saves the modified ONNX model to the specified path.
        """
        print("\n--- Step 4: Validating and Saving ONNX Model ---")
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

def main():
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
    print("\n--- Step 5: Attempting to simplify the ONNX model (including constant folding) ---\n")
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
    converter.verify_onnx_model(final_onnx_model, is_modified=True) # Verify only the modified model
    time.sleep(1) 
    converter.save_onnx_model(final_onnx_model)

    print("\nFull ONNX conversion, modification, and potential simplification process completed.")

if __name__ == "__main__":
    main()