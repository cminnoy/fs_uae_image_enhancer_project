import torch
import torch.nn as nn
import time
import sys

# Line numbers are for reference purposes in this explanation; they are not part of the script itself.

# (Lines 1-25: Conv2dBenchmark Class - unchanged from previous version for consistency)
class Conv2dBenchmark(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

# (Lines 27-142: BenchmarkRunner Class - significantly enhanced)
class BenchmarkRunner:
    def __init__(self, input_shape: tuple, num_warmup_runs: int = 20, num_benchmark_runs: int = 100):
        self.input_shape = input_shape
        self.num_warmup_runs = num_warmup_runs
        self.num_benchmark_runs = num_benchmark_runs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Benchmarking on device: {self.device}")

    def _calculate_flops_single_conv(self, model: nn.Module, input_tensor: torch.Tensor) -> float:
        """
        Calculates theoretical GFLOPs for a single Conv2d layer within the model.
        """
        if not hasattr(model, 'conv') or not isinstance(model.conv, nn.Conv2d):
            raise ValueError("Model must contain an 'nn.Conv2d' named 'conv' for FLOPs calculation.")

        conv_layer = model.conv

        batch_size, in_channels, H_in, W_in = input_tensor.shape
        out_channels = conv_layer.out_channels
        kernel_h, kernel_w = conv_layer.kernel_size
        padding_h, padding_w = conv_layer.padding
        stride_h, stride_w = conv_layer.stride

        H_out = (H_in + 2 * padding_h - kernel_h) // stride_h + 1
        W_out = (W_in + 2 * padding_w - kernel_w) // stride_w + 1

        flops_per_output_pixel = kernel_h * kernel_w * in_channels * out_channels
        total_flops = float(batch_size * H_out * W_out * flops_per_output_pixel * 2)
        return total_flops / 1e9 # Return in GFLOPs

    def benchmark_layer(self, model_class: type(nn.Module), in_channels: int, out_channels: int,
                        kernel_size: int, dtype: torch.dtype) -> tuple:
        """
        Benchmarks a single convolutional layer and returns average time and GFLOPs.
        """
        # Create model and dummy input on the correct device and dtype
        model = model_class(in_channels, out_channels, kernel_size).to(self.device).to(dtype)
        dummy_input = torch.randn(self.input_shape, device=self.device, dtype=dtype)

        # Warm-up runs to stabilize performance
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        with torch.no_grad():
            for _ in range(self.num_warmup_runs):
                _ = model(dummy_input)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # Benchmark runs for accurate measurement
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(self.num_benchmark_runs):
                _ = model(dummy_input)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        avg_time_ms = (end_time - start_time) / self.num_benchmark_runs * 1000
        gflops = self._calculate_flops_single_conv(model, dummy_input)

        return avg_time_ms, gflops

    def find_channels_for_target_time(self, model_class: type(nn.Module), in_channels: int,
                                       kernel_size: int, dtype: torch.dtype,
                                       target_time_ms: float, max_channels_to_search: int = 1024,
                                       channel_step: int = 1, tolerance_ms: float = 0.005) -> tuple:
        """
        Performs a linear search to find the number of output channels whose benchmark time
        is closest to the target_time_ms.
        """
        print(f"\nSearching for channels for Kernel: {kernel_size}x{kernel_size}, Dtype: {str(dtype).split('.')[-1]} to match target time {target_time_ms:.4f} ms...")
        best_channels = 0
        min_diff = float('inf')
        found_time = None
        
        # We assume that time generally increases with channels, so we can stop if we significantly overshoot.
        # This makes the search more efficient.
        overshoot_threshold_factor = 1.5 # If time is 50% more than target and getting worse, stop.
        
        # Start search from a reasonable low channel count (e.g., 1 or 8)
        for out_ch in range(1, max_channels_to_search + 1, channel_step):
            try:
                avg_time, _ = self.benchmark_layer(model_class, in_channels, out_ch, kernel_size, dtype)
                
                current_diff = abs(avg_time - target_time_ms)
                
                # Update best found channels if current is closer
                if current_diff < min_diff:
                    min_diff = current_diff
                    best_channels = out_ch
                    found_time = avg_time
                
                # Early stopping condition: If we've clearly passed the target and moved far away,
                # or if an exact match is found.
                if current_diff <= tolerance_ms: # If we are within tolerance, we found a good match
                    best_channels = out_ch
                    found_time = avg_time
                    break
                
                # Heuristic: If we have already exceeded the target time AND the current time is significantly higher
                # than the previous best time, it's likely we passed the optimal point.
                if avg_time > target_time_ms * overshoot_threshold_factor and best_channels > 0 and current_diff > min_diff:
                    print(f"  Early stopping for {kernel_size}x{kernel_size}: time {avg_time:.4f} ms exceeded target and diverged.")
                    break

            except torch.cuda.OutOfMemoryError:
                print(f"  CUDA Out of Memory for {kernel_size}x{kernel_size} with {out_ch} channels. Stopping search for this kernel.")
                break
            except Exception as e:
                print(f"  Error benchmarking {kernel_size}x{kernel_size} with {out_ch} channels: {e}. Stopping search.")
                break

        if best_channels == 0 and found_time is None: # If no suitable channels found
            print(f"  Could not find channels within reasonable range for {kernel_size}x{kernel_size} to match target time {target_time_ms:.4f} ms.")
            return 0, 0.0 # Return 0 channels and 0 time
        else:
            print(f"  Found approx. {best_channels} channels (Time: {found_time:.4f} ms) for {kernel_size}x{kernel_size} closest to {target_time_ms:.4f} ms.")
            return best_channels, found_time

    def run_search_phase(self, target_total_channels: int = 256):
        """
        Orchestrates the search for exact channel counts for each kernel size
        to match a common time budget, then scales them to sum to the target_total_channels.
        """
        # Phase 1: Determine the target time by benchmarking a reference configuration.
        # Based on your previous benchmark data, 1x1, 16 channels, float16 was among the fastest.
        # We will use this as our initial reference to establish the target time.
        reference_kernel_size = 1
        reference_out_channels = 16 # As per your fastest previous float16 1x1 entry
        reference_dtype = torch.float16

        print("\n--- Phase 1: Determining Target Time ---")
        print(f"Benchmarking reference (Kernel: {reference_kernel_size}x{reference_kernel_size}, Channels: {reference_out_channels}, Dtype: {str(reference_dtype).split('.')[-1]})...")
        target_time_ms, _ = self.benchmark_layer(Conv2dBenchmark,
                                                   self.input_shape[1], # Input channels (3)
                                                   reference_out_channels,
                                                   reference_kernel_size,
                                                   reference_dtype)
        print(f"Established Target Time for individual pathways: {target_time_ms:.4f} ms")

        # Phase 2: Search for channels for each kernel size to match the target time
        found_channel_configs = {}
        kernel_sizes_to_search = [1, 2, 3, 4, 5, 7] # All kernel sizes
        dtype_for_search = torch.float16 # User specified float16 for this phase

        print("\n--- Phase 2: Searching for Channels per Kernel Size to Match Target Time ---")
        # Define search parameters for precision and efficiency
        max_search_channels = 1024 # Maximum channels to test for any given kernel (adjust based on GPU memory)
        channel_increment_step = 1 # Fine-grained search (1 channel at a time)
        time_match_tolerance = 0.005 # milliseconds (e.g., within 0.005 ms of target)

        for k_size in kernel_sizes_to_search:
            channels, actual_time = self.find_channels_for_target_time(Conv2dBenchmark,
                                                                        self.input_shape[1], # Input channels (3)
                                                                        k_size,
                                                                        dtype_for_search,
                                                                        target_time_ms,
                                                                        max_channels_to_search=max_search_channels,
                                                                        channel_step=channel_increment_step,
                                                                        tolerance_ms=time_match_tolerance)
            if channels > 0: # Only store valid findings
                found_channel_configs[k_size] = {'channels': channels, 'time_ms': actual_time}

        # Phase 3: Scale the found channels to sum up to the target_total_channels (256)
        print(f"\n--- Phase 3: Scaling Channels to Sum to Total {target_total_channels} Output Channels ---")
        
        # Calculate the sum of the 'equal time' channels found
        total_relative_channels = sum(config['channels'] for config in found_channel_configs.values())

        if total_relative_channels == 0:
            print("No suitable channel configurations found to scale. Please check search parameters or target time.")
            return

        scaled_channel_distribution = {}
        # Calculate initial scaled values
        for k_size, config in found_channel_configs.items():
            scaled_channels = (config['channels'] / total_relative_channels) * target_total_channels
            # Store float value first to maintain precision for rounding
            scaled_channel_distribution[k_size] = scaled_channels

        # Round and distribute any remaining channels due to rounding differences
        rounded_sum = 0
        # Convert to integer and store difference from original float value
        for k_size in scaled_channel_distribution:
            initial_rounded_ch = int(scaled_channel_distribution[k_size])
            scaled_channel_distribution[k_size] = {'rounded': initial_rounded_ch, 'fraction': scaled_channel_distribution[k_size] - initial_rounded_ch}
            rounded_sum += initial_rounded_ch
        
        remaining_channels = target_total_channels - rounded_sum

        # Distribute remaining channels to those with the largest fractional part first
        # This ensures the sum is exactly target_total_channels and distributes rounding errors smartly
        sorted_by_fraction = sorted(scaled_channel_distribution.items(), key=lambda item: item[1]['fraction'], reverse=True)
        
        for k_size, _ in sorted_by_fraction:
            if remaining_channels <= 0:
                break
            scaled_channel_distribution[k_size]['rounded'] += 1
            remaining_channels -= 1

        # Final display of the recommended distribution
        print(f"Goal: Total {target_total_channels} output channels in {str(dtype_for_search).split('.')[-1]} across parallel paths, each aiming for ~{target_time_ms:.4f} ms.")
        print("\nRecommended Channel Distribution for Parallel Paths:")
        print("{:<12} {:<10} {:<15} {:<10}".format("Kernel Size", "Channels", "Expected Time (ms)", "Expected GFLOPs"))
        print("-" * 60)
        
        final_sum_channels = 0
        for k_size in kernel_sizes_to_search: # Print in original kernel size order
            if k_size in scaled_channel_distribution:
                assigned_channels = scaled_channel_distribution[k_size]['rounded']
                final_sum_channels += assigned_channels
                
                # Re-benchmark with the assigned channels to show their actual expected performance
                # This gives a final verification of the per-path timing
                final_path_time, final_path_gflops = self.benchmark_layer(Conv2dBenchmark,
                                                                          self.input_shape[1],
                                                                          assigned_channels,
                                                                          k_size,
                                                                          dtype_for_search)
                print("{:<12} {:<10} {:<15.4f} {:<10.3f}".format(
                    f"{k_size}x{k_size}",
                    assigned_channels,
                    final_path_time,
                    final_path_gflops
                ))
            else:
                print(f"{k_size}x{k_size} : No suitable channels found or included in scaling.")
        
        print(f"\nTotal Sum of Recommended Output Channels: {final_sum_channels}")
        print("\nThese recommended channel counts aim for roughly equal per-path latency, summing to the target total channels.")
        print("Remember that exact timings can fluctuate; a final verification on your specific AMD hardware after implementation is highly recommended.")


# Main execution block
if __name__ == "__main__":
    # Input tensor is 576 H and 752 W, with 3 input channels (e.g., RGB image).
    # Batch size is 1 for typical single-image inference.
    input_tensor_shape = (1, 3, 576, 752)

    # Instantiate the benchmark runner
    runner = BenchmarkRunner(input_tensor_shape)

    # Execute the search phase for a total of n output channels, all in float16
    runner.run_search_phase(target_total_channels=64)