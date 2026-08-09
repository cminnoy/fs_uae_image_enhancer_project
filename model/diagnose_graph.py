import onnx
import numpy as np
import onnxruntime as ort
from pathlib import Path

def diagnose_onnx_model(model_path, input_tensor):
    model = onnx.load(model_path)
    graph = model.graph

    # Collect all intermediate tensor names so we can inspect them
    all_tensor_names = set()
    for node in graph.node:
        for out_name in node.output:
            if out_name:
                all_tensor_names.add(out_name)

    print(f"All tensor names: {all_tensor_names}")

    # To avoid memory/performance explosion, we can add outputs selectively, 
    # or test blocks. For a full sweep, let's create a session and inspect graph nodes.
    # Alternatively, you can target specific layers by name substring (e.g., "Pow", "Silu", "mamba")

    print(f"Total nodes in graph: {len(graph.node)}")

    # Let's inspect nodes that contain activation or math operations prone to FP16 overflow
    target_ops = {"Pow", "Div", "Mul", "Clip", "Gemm", "MatMul"}

    for i, node in enumerate(graph.node):
        if node.op_type in target_ops or "mamba" in node.name.lower():
            # Temporarily mark this node's outputs as graph outputs to inspect them
            added_outputs = []
            for out_name in node.output:
                if out_name and not any(o.name == out_name for o in graph.output):
                    # Find value info or create dummy value info
                    val_info = onnx.helper.make_tensor_value_info(out_name, onnx.TensorProto.DataType.FLOAT16, None)
                    graph.output.append(val_info)
                    added_outputs.append(out_name)

            # Save temporary debug model
            debug_model_path = "debug_temp.onnx"
            onnx.save(model, debug_model_path)

            try:
                # Run with MIGraphX (or CUDA)
                options = ort.SessionOptions()
                providers = ['MIGraphXExecutionProvider', 'CPUExecutionProvider']
                session = ort.InferenceSession(debug_model_path, sess_options=options, providers=providers)

                input_name = session.get_inputs()[0].name
                results = session.run(None, {input_name: input_tensor})

                # Map results back to outputs
                output_names = [o.name for o in session.get_outputs()]
                for name in added_outputs:
                    idx = output_names.index(name)
                    val = results[idx]
                    has_nan = np.isnan(val).any()
                    has_inf = np.isinf(val).any()
                    min_val = np.min(val) if val.size > 0 else 0
                    max_val = np.max(val) if val.size > 0 else 0

                    if has_nan or has_inf or (max_val == 0 and min_val == 0):
                        print(f"[!] ANOMALY DETECTED at Node {i} | Name: {node.name} | Op: {node.op_type} | Output: {name}")
                        print(f"    Min: {min_val}, Max: {max_val}, NaN: {has_nan}, Inf: {has_inf}")
            except Exception as e:
                # Some nodes might fail if shape inference complains when exposed as outputs
                pass
            finally:
                # Clean up added outputs for the next iteration
                for _ in added_outputs:
                    graph.output.pop()

            # Revert model graph structure for next loop
            model = onnx.load(model_path)
            graph = model.graph

if __name__ == "__main__":
    # Load sample input tensor [1, 576, 752, 4] uint8
    dummy_input = np.random.randint(0, 256, (1, 576, 752, 4), dtype=np.uint8)
    diagnose_onnx_model("/workspaces/fs-uae/upscaler/model_fp16_2.onnx", dummy_input)