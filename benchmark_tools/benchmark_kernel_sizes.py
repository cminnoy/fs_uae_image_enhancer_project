import torch
import torch.nn as nn
import time
import sys

# Define the Conv2d Benchmark Module
# This class encapsulates a single Conv2d layer for isolated benchmarking.
class Conv2dBenchmark(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        """
        Initializes a Conv2d layer with specified channels and kernel size.
        Padding is automatically calculated to maintain spatial dimensions.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel (e.g., 1, 3, 5, 7).
        """
        super().__init__()
        # Calculate padding to ensure the output spatial dimensions match the input.
        # This is crucial for maintaining consistent input/output sizes in subsequent layers.
        padding = (kernel_size - 1) // 2
        # Using bias=False as per your BasicModel's current practice with BatchNorm.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Conv2d layer.
        """
        return self.conv(x)

# Define the Benchmark Runner Class
# This class handles the setup, execution, and reporting of the benchmarks.
class BenchmarkRunner:
    def __init__(self, input_shape: tuple, num_warmup_runs: int = 20, num_benchmark_runs: int = 100):
        """
        Initializes the benchmark runner.

        Args:
            input_shape (tuple): The shape of the input tensor (Batch, C_in, H, W).
            num_warmup_runs (int): Number of warm-up runs to prime the GPU/CPU.
            num_benchmark_runs (int): Number of actual benchmark runs to average over.
        """
        self.input_shape = input_shape
        self.num_warmup_runs = num_warmup_runs
        self.num_benchmark_runs = num_benchmark_runs
        # Attempt to use CUDA (GPU) if available, otherwise fall back to CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Benchmarking on device: {self.device}")

    def _calculate_flops_single_conv(self, model: nn.Module, input_tensor: torch.Tensor) -> float:
        """
        Calculates theoretical GFLOPs for a single Conv2d layer.
        This provides a standardized measure of computational complexity.
        """
        # Ensure the model is an instance of Conv2dBenchmark or similar wrapping nn.Conv2d
        if not hasattr(model, 'conv') or not isinstance(model.conv, nn.Conv2d):
            raise ValueError("Model must contain an 'nn.Conv2d' named 'conv' for FLOPs calculation.")

        conv_layer = model.conv

        batch_size, in_channels, H_in, W_in = input_tensor.shape
        out_channels = conv_layer.out_channels
        kernel_h, kernel_w = conv_layer.kernel_size
        padding_h, padding_w = conv_layer.padding
        stride_h, stride_w = conv_layer.stride

        # Calculate output spatial dimensions
        H_out = (H_in + 2 * padding_h - kernel_h) // stride_h + 1
        W_out = (W_in + 2 * padding_w - kernel_w) // stride_w + 1

        # FLOPs formula: (Output Pixels) * (Kernel Size) * (Input Channels) * (Output Channels) * 2 (for mul-add)
        # We multiply by 2 because each MAC (Multiply-Accumulate) operation is typically counted as 2 FLOPs.
        flops_per_output_pixel = kernel_h * kernel_w * in_channels * out_channels
        total_flops = float(batch_size * H_out * W_out * flops_per_output_pixel * 2)
        return total_flops / 1e9 # Return in GFLOPs (Giga Floating Point Operations)

    def benchmark_layer(self, model_class: type(nn.Module), in_channels: int, out_channels: int,
                        kernel_size: int, dtype: torch.dtype) -> tuple:
        """
        Benchmarks a single convolutional layer.

        Args:
            model_class (type): The class of the model to benchmark (e.g., Conv2dBenchmark).
            in_channels (int): Input channels for the layer.
            out_channels (int): Output channels for the layer.
            kernel_size (int): Kernel size for the layer.
            dtype (torch.dtype): Data type for benchmarking (torch.float32 or torch.float16).

        Returns:
            tuple: A tuple containing (average_time_ms_per_inference, gflops_of_layer).
        """
        # Instantiate the model and move it to the device and specified dtype.
        model = model_class(in_channels, out_channels, kernel_size).to(self.device).to(dtype)
        # Create a dummy input tensor with random data on the correct device and dtype.
        dummy_input = torch.randn(self.input_shape, device=self.device, dtype=dtype)

        # Warm-up runs: These runs help to "warm up" the GPU and allocate necessary memory,
        # ensuring that subsequent measurements are more representative of steady-state performance.
        if self.device.type == 'cuda':
            torch.cuda.synchronize() # Ensure any previous ops are done before warm-up
        with torch.no_grad(): # Disable gradient calculations for inference, saving memory and time.
            for _ in range(self.num_warmup_runs):
                _ = model(dummy_input)
        if self.device.type == 'cuda':
            torch.cuda.synchronize() # Synchronize after warm-up

        # Benchmark runs: Measure the actual execution time.
        start_time = time.perf_counter() # Use perf_counter for higher resolution timing
        with torch.no_grad():
            for _ in range(self.num_benchmark_runs):
                _ = model(dummy_input)
        if self.device.type == 'cuda':
            torch.cuda.synchronize() # Synchronize before stopping timer for accurate GPU timing
        end_time = time.perf_counter()

        # Calculate average time per inference in milliseconds.
        avg_time_ms = (end_time - start_time) / self.num_benchmark_runs * 1000
        # Calculate GFLOPs for the layer.
        gflops = self._calculate_flops_single_conv(model, dummy_input)

        return avg_time_ms, gflops

    def run_all_benchmarks(self, kernel_sizes: list = None,
                           output_channel_ranges: list = None):
        """
        Runs benchmarks for all specified kernel sizes, data types, and output channel counts.

        Args:
            kernel_sizes (list): A list of kernel sizes to benchmark (e.g., [1, 2, 3, 5, 7]).
                                  Defaults to [1, 2, 3, 5, 7] if None.
            output_channel_ranges (list): A list of output channel counts to test for each kernel.
                                           Defaults to [16, 32, 64, 96, 128, 192, 256, 384, 512] if None.
        Returns:
            list: A list of dictionaries, each containing benchmark results.
        """
        if kernel_sizes is None:
            kernel_sizes = [1, 2, 3, 4, 5, 7]
        if output_channel_ranges is None:
            output_channel_ranges = [16, 32, 64, 96, 128, 192, 256, 384, 512] # A good range to explore

        results = []
        print("\n--- Starting Benchmarks ---")

        # Iterate through each kernel size, then each data type, then each channel count
        for k_size in kernel_sizes:
            for dtype in [torch.float32, torch.float16]:
                for out_ch in output_channel_ranges:
                    print(f"Benchmarking Kernel: {k_size}x{k_size}, Dtype: {str(dtype).split('.')[-1]}, Output Channels: {out_ch}...")
                    avg_time, gflops = self.benchmark_layer(Conv2dBenchmark,
                                                            self.input_shape[1], # Input channels (3)
                                                            out_ch,
                                                            k_size,
                                                            dtype)
                    results.append({
                        'kernel_size': f"{k_size}x{k_size}",
                        'output_channels': out_ch,
                        'dtype': str(dtype).split('.')[-1],
                        'avg_time_ms': avg_time,
                        'gflops': gflops
                    })
                    print(f"  Avg Time: {avg_time:.4f} ms/inference, GFLOPs: {gflops:.3f}")

        print("\n--- Benchmark Summary ---")
        print(f"Input Shape: {self.input_shape}")
        # Format for clear output
        print("{:<12} {:<10} {:<10} {:<15} {:<10}".format("Kernel Size", "Channels", "Dtype", "Avg Time (ms)", "GFLOPs"))
        print("-" * 60)
        for res in results:
            print("{:<12} {:<10} {:<10} {:<15.4f} {:<10.3f}".format(
                res['kernel_size'],
                res['output_channels'],
                res['dtype'],
                res['avg_time_ms'],
                res['gflops']
            ))

        return results

# Main execution block
if __name__ == "__main__":
    # Your specified input tensor dimensions: 576 H and 752 W.
    # As it's for the first layer, input channels are 3 (e.g., RGB image).
    # Batch size is set to 1 for typical single-image inference.
    input_tensor_shape = (1, 3, 576, 752)

    # Instantiate the benchmark runner
    runner = BenchmarkRunner(input_tensor_shape)

    # Define the output channel ranges to test.
    # You can adjust this list to explore more densely or sparingly.
    channels_to_test = [16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024] # Added more channels

    # Run the benchmarks
    benchmark_results = runner.run_all_benchmarks(output_channel_ranges=channels_to_test)