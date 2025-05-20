import torch
import torch.nn as nn
import time
import math

# --- Configuration ---
NUM_WARMUP = 20   # Number of warm-up runs to prime the GPU
NUM_REPEATS = 200 # Number of benchmark runs to average over

# --- Activation Functions to Test ---
# Using nn.Module versions as they handle state and are used in networks
ACTIVATION_FUNCTIONS = {
    "ReLU": nn.ReLU(),
    "LeakyReLU": nn.LeakyReLU(),
    "PReLU": nn.PReLU(), # Will instantiate with correct params based on shape
    "GELU": nn.GELU(),
    "SiLU (Swish)": nn.SiLU(),
    "Mish": nn.Mish(),
    "Hardtanh": nn.Hardtanh(),
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh(),
    "Hardswish": nn.Hardswish(),
    "Softplus": nn.Softplus(), # Uses default beta=1.0, threshold=20
    "Softsign": nn.Softsign(),
    # Add or remove activations as needed
}

# --- Input Tensor Shapes to Test ---
# Representing common shapes from CNNs (NCHW) and Linear layers (NL)
INPUT_SHAPES = [
    (64, 128, 56, 56),  # Example: batch=64, channels=128, HxW=56x56
    (128, 256, 28, 28), # Example: batch=128, channels=256, HxW=28x28
    (256, 512, 14, 14), # Example: batch=256, channels=512, HxW=14x14
    (128, 1024),       # Example: batch=128, features=1024
    (256, 2048),       # Example: batch=256, features=2048
]

# --- Data Types to Test ---
DTYPES_TO_TEST = [torch.float32]
if torch.cuda.is_available():
    DTYPES_TO_TEST.append(torch.float16) # Test half precision on GPU

# --- Benchmark Class ---
class ActivationBenchmark:
    # Line 48
    def __init__(self, num_warmup: int = NUM_WARMUP, num_repeats: int = NUM_REPEATS):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == 'cpu':
            print("Warning: CUDA not available. Benchmarking on CPU will not reflect GPU performance.")
            print("GPU benchmarking uses torch.cuda.Event for accurate timing.")
            print("CPU benchmarking uses time.perf_counter (less precise for very fast ops).")

        self.num_warmup = num_warmup
        self.num_repeats = num_repeats

    # Line 60
    def benchmark(self, activation_module: nn.Module, input_shape: tuple, dtype: torch.dtype = torch.float32):
        """
        Benchmarks the forward and backward pass time for a given activation function.
        """
        # Create a dummy input tensor
        x = torch.randn(input_shape, device=self.device, dtype=dtype)

        # Ensure module is on the correct device and dtype
        activation_module.to(self.device).type(dtype)

        # Warm-up runs
        # This helps the GPU allocate resources and settle into a stable clock speed
        with torch.no_grad(): # Warmup typically doesn't need gradients
            for _ in range(self.num_warmup):
                _ = activation_module(x)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize() # Wait for GPU operations to complete

        # Benchmarking runs
        forward_times = []
        backward_times = []

        start_event = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        end_event = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None

        # Need retain_graph=True because we call backward multiple times on the same graph structure
        # In a real training loop, the graph is built and destroyed per iteration.
        # This is a synthetic benchmark; retain_graph is needed for repeat backward calls.
        retain_graph_flag = True if self.num_repeats > 1 else False


        for _ in range(self.num_repeats):
            # --- Forward Pass ---
            if self.device.type == 'cuda':
                start_event.record()
            start_time_cpu = time.perf_counter() if self.device.type == 'cpu' else None

            x.requires_grad_(True) # Enable gradient tracking for the input
            output = activation_module(x)

            if self.device.type == 'cuda':
                end_event.record()
                torch.cuda.synchronize()
                forward_times.append(start_event.elapsed_time(end_event)) # elapsed_time is in ms
            else:
                end_time_cpu = time.perf_counter()
                forward_times.append((end_time_cpu - start_time_cpu) * 1000) # Convert to ms

            # --- Backward Pass ---
            # Create a dummy gradient for the output to backpropagate
            dummy_grad = torch.ones_like(output, device=self.device, dtype=dtype)
            # output.retain_grad() # Optional: Keep gradient of output itself


            if self.device.type == 'cuda':
                 start_event.record()
            start_time_cpu = time.perf_counter() if self.device.type == 'cpu' else None

            # Perform backward pass
            output.backward(dummy_grad, retain_graph=retain_graph_flag)

            if self.device.type == 'cuda':
                end_event.record()
                torch.cuda.synchronize()
                backward_times.append(start_event.elapsed_time(end_event))
            else:
                end_time_cpu = time.perf_counter()
                backward_times.append((end_time_cpu - start_time_cpu) * 1000) # Convert to ms

            # Detach input gradients to free memory and prepare for next iteration
            x.requires_grad_(False)
            if x.grad is not None:
                x.grad.zero_() # Clear gradients

        avg_forward_time = sum(forward_times) / self.num_repeats
        avg_backward_time = sum(backward_times) / self.num_repeats

        return avg_forward_time, avg_backward_time

# --- Main Execution ---
# Line 150
if __name__ == "__main__":
    benchmark_runner = ActivationBenchmark()

    print("--- Activation Function Performance Benchmark ---")
    print(f"Running on device: {benchmark_runner.device}")
    print(f"Warm-up runs: {NUM_WARMUP}, Benchmark repeats: {NUM_REPEATS}")
    print("-" * 60)

    for dtype in DTYPES_TO_TEST:
        print(f"\nBenchmarking Data Type: {dtype}")
        print("=" * 60)

        for shape in INPUT_SHAPES:
            print(f"\nBenchmarking Input Shape: {shape}")
            print("-" * 30)
            results = {}
            # Sort activations alphabetically for consistent output order
            sorted_activation_names = sorted(ACTIVATION_FUNCTIONS.keys())

            for name in sorted_activation_names:
                activation_module_template = ACTIVATION_FUNCTIONS[name]
                try:
                    # Re-instantiate activation module for each run to ensure clean state
                    # Handle modules needing specific parameters based on shape
                    if name == "PReLU":
                        # PReLU needs num_parameters = number of channels/features
                        if len(shape) == 4: # NCHW
                             num_params = shape[1]
                        elif len(shape) == 2: # NL
                             num_params = shape[1]
                        elif len(shape) == 1: # N
                             num_params = shape[0]
                        else: # Handle other dimensions if necessary, default to 1?
                            num_params = 1
                            print(f"  Warning: PReLU shape {shape} unexpected, using num_parameters=1")
                        activation_module = nn.PReLU(num_parameters=num_params)
                    elif name == "Hardshrink":
                        activation_module = nn.Hardshrink(lambd=0.5) # Using a default lambda
                    elif name == "Softplus":
                         activation_module = nn.Softplus() # Using default beta=1, threshold=20
                    # Add other special cases requiring parameters here
                    else:
                        # For modules without parameters, create a new instance of the class
                        activation_module = activation_module_template.__class__()

                    # Run the benchmark for this activation, shape, and dtype
                    fwd_time, bwd_time = benchmark_runner.benchmark(activation_module, shape, dtype=dtype)
                    results[name] = (fwd_time, bwd_time)
                    print(f"  {name:<15}: Forward: {fwd_time:.4f} ms, Backward: {bwd_time:.4f} ms")
                except Exception as e:
                    print(f"  Benchmarking {name:<15} failed for shape {shape}, dtype {dtype}: {e}")
                    results[name] = (float('nan'), float('nan')) # Indicate failure


            # Optional: Print sorted results for this shape/dtype
            # print("\n  Sorted by Total Time (Forward + Backward):")
            # sorted_results = sorted(results.items(), key=lambda item: item[1][0] + item[1][1] if not math.isnan(item[1][0]) else float('inf'))
            # for name, times in sorted_results:
            #      if not math.isnan(times[0]):
            #          print(f"  {name:<15}: Total: {times[0] + times[1]:.4f} ms")
            #      else:
            #          print(f"  {name:<15}: Failed")

        print("\n" + "=" * 60)

    print("\nBenchmark Complete.")
    print("Note: Results can vary based on GPU load, driver version, PyTorch version, etc.")