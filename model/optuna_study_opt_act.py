import os, time, argparse, warnings, sys
from typing import Tuple
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

import optuna
from optuna.pruners import MedianPruner
from model_conv6 import Model # Ensure Model class is imported for instantiation in caching
from srdataset import SRDataset, gather_all_samples_from_directory


# --- Global Performance Cache ---
# This dictionary will store (model_config_key -> measured_fps)
# It will be loaded from and saved to a JSON file.
performance_cache = {}
PERFORMANCE_CACHE_FILE = "model_performance_cache.json"

def load_performance_cache():
    global performance_cache
    if os.path.exists(PERFORMANCE_CACHE_FILE):
        with open(PERFORMANCE_CACHE_FILE, 'r') as f:
            try:
                performance_cache = json.load(f)
                print(f"Loaded {len(performance_cache)} entries from performance cache.")
            except json.JSONDecodeError:
                print("Warning: Could not decode performance cache file. Starting with empty cache.")
                performance_cache = {}
    else:
        print("No existing performance cache found. Starting with empty cache.")

def save_performance_cache():
    global performance_cache
    with open(PERFORMANCE_CACHE_FILE, 'w') as f:
        json.dump(performance_cache, f, indent=4)
    print(f"Performance cache saved to {PERFORMANCE_CACHE_FILE} ({len(performance_cache)} entries).")

# --- Reusable Performance Measurement Function ---
def measure_performance(model: nn.Module, device: torch.device, input_size: Tuple[int, int, int, int] = (1, 3, 576, 752), warmup_iterations: int = 20, measure_duration: int = 20) -> float:
    """
    Measures the sustained FPS of the model on a dummy input tensor.

    Args:
        model: The PyTorch model to measure.
        device: The device to run the model on (cpu or cuda).
        input_size: The size of the dummy input tensor (Batch, Channels, Height, Width).
        warmup_iterations: Number of warm-up forward passes.
        measure_duration: Duration in seconds to measure FPS over.

    Returns:
        The measured FPS. Returns 0 if measurement fails.
    """
    model.eval()
    model.to(device)
    try:
        # Check if torch.compile is available (requires PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            torch.compiler.reset()
            compiled_model = torch.compile(model, mode="default", fullgraph=True)
            model = compiled_model
            print("Model compiled successfully for performance test.")
        else:
            print("Warning: torch.compile not available. Skipping compilation.")

        dummy_input = torch.rand(input_size, dtype=torch.float32).to(device) # Use float16 for AMP context

        # Warm-up
        print("Starting performance test warm-up (GPU)...")
        if device.type == 'cuda':
            with autocast(device_type='cuda', enabled=False): # Explicitly specify device_type and enabled
                with torch.no_grad():
                    for _ in range(warmup_iterations):
                        _ = model(dummy_input)
            print("Warm-up finished.")
        else:
            print("Starting performance test warm-up (CPU)...")
            with torch.no_grad():
                for _ in range(warmup_iterations):
                    _ = model(dummy_input)
            print("Warm-up finished (CPU).")

        # Measure FPS
        start_time = time.time()
        num_iterations = 0
        
        print("Measuring FPS...")
        if device.type == 'cuda':
            with autocast(device_type='cuda', enabled=False): # Explicitly specify device_type and enabled
                with torch.no_grad():
                    while time.time() - start_time < measure_duration:
                        _ = model(dummy_input)
                        num_iterations += 1
        else:
            with torch.no_grad():
                while time.time() - start_time < measure_duration:
                    _ = model(dummy_input)
                    num_iterations += 1

        elapsed_time = time.time() - start_time
        fps = num_iterations / elapsed_time
        print(f"Performance test finished. Measured FPS: {fps:.2f}")
        return fps
    except Exception as e:
        print(f"Error during performance measurement: {e}")
        return 0 # Indicate failure

# --- Optuna Objective Function ---
def objective(trial: optuna.Trial) -> float:
    global args # Access the command-line arguments

    # Hyperparameters to tune
    # Define a set of parameters that define the unique model architecture/performance
    # This will be used to create the cache key
    params_for_perf_key = {}
    
    # Layer 1
  # params_for_perf_key['layer1_out_channels'] = trial.suggest_int('layer1_out_channels', 36, 56, step=2)
  # params_for_perf_key['layer1_kernel_size'] = trial.suggest_categorical('layer1_kernel_size', [3, 5])
    params_for_perf_key['layer1_act1'] = trial.suggest_categorical('layer1_act1', ['identity', 'tanh', 'telu', 'sinlu', 'mish', 'silu'])
    params_for_perf_key['layer1_act2'] = trial.suggest_categorical('layer1_act2', ['identity', 'relu', 'leaky_relu', 'biased_relu', 'biased_prelu', 'prelu', 'relu6'])

    # Layer 2
  # params_for_perf_key['layer2_out_channels'] = trial.suggest_int('layer2_out_channels', 36, 72, step=2)
  # params_for_perf_key['layer2_kernel_size'] = trial.suggest_categorical('layer2_kernel_size', [3, 5])
    params_for_perf_key['layer2_act1'] = trial.suggest_categorical('layer2_act1', ['identity', 'tanh', 'telu', 'sinlu', 'mish', 'silu'])
    params_for_perf_key['layer2_act2'] = trial.suggest_categorical('layer2_act2', ['identity', 'relu', 'leaky_relu', 'biased_relu', 'biased_prelu', 'prelu', 'relu6'])
    params_for_perf_key['layer2_act3'] = trial.suggest_categorical('layer2_act3', ['identity', 'tanh', 'telu', 'sinlu', 'mish', 'silu'])
    params_for_perf_key['layer2_act4'] = trial.suggest_categorical('layer2_act4', ['identity', 'relu', 'leaky_relu', 'biased_relu', 'biased_prelu', 'prelu', 'relu6'])

    # Layer 4
  # params_for_perf_key['layer4_out_channels'] = trial.suggest_int('layer4_out_channels', 36, 72, step=2)
  #  params_for_perf_key['layer4_kernel_size'] = trial.suggest_categorical('layer4_kernel_size', [3, 5])
    params_for_perf_key['layer4_act1'] = trial.suggest_categorical('layer4_act1', ['identity', 'tanh', 'telu', 'sinlu', 'mish', 'silu'])
    params_for_perf_key['layer4_act2'] = trial.suggest_categorical('layer4_act2', ['identity', 'relu', 'leaky_relu', 'biased_relu', 'biased_prelu', 'prelu', 'relu6'])
    params_for_perf_key['layer4_act3'] = trial.suggest_categorical('layer4_act3', ['identity', 'tanh', 'telu', 'sinlu', 'mish', 'silu'])
    params_for_perf_key['layer4_act4'] = trial.suggest_categorical('layer4_act4', ['identity', 'relu', 'leaky_relu', 'biased_relu', 'biased_prelu', 'prelu', 'relu6'])

    # Layer 6
  # params_for_perf_key['layer6_out_channels'] = trial.suggest_int('layer6_out_channels', 36, 72, step=2)
  # params_for_perf_key['layer6_kernel_size'] = trial.suggest_categorical('layer6_kernel_size', [3, 5])
    params_for_perf_key['layer6_act1'] = trial.suggest_categorical('layer6_act1', ['identity', 'tanh', 'telu', 'sinlu', 'mish', 'silu'])
    params_for_perf_key['layer6_act2'] = trial.suggest_categorical('layer6_act2', ['identity', 'relu', 'leaky_relu', 'biased_relu', 'biased_prelu', 'prelu', 'relu6'])

    # Layer 7
  # params_for_perf_key['layer7_kernel_size'] = trial.suggest_categorical('layer7_kernel_size', [3, 5])
    params_for_perf_key['layer7_act1'] = trial.suggest_categorical('layer7_act1', ['identity', 'tanh', 'telu', 'sinlu', 'mish', 'silu'])
    params_for_perf_key['layer7_act2'] = trial.suggest_categorical('layer7_act2', ['identity', 'relu', 'leaky_relu', 'biased_relu', 'biased_prelu', 'prelu', 'relu6'])

    for layer_act_name in['layer1_act2', 'layer2_act2', 'layer2_act4', 'layer4_act2', 'layer4_act4', 'layer6_act2', 'layer7_act2']:  
        if params_for_perf_key[layer_act_name] == 'leaky_relu':
            params_for_perf_key[f'{layer_act_name}_params'] = {} 
            params_for_perf_key[f'{layer_act_name}_params']['negative_slope']  = trial.suggest_float(f'{layer_act_name}_negative_slope', low=0.001, high=0.5)
        elif params_for_perf_key[layer_act_name] in ['prelu', 'biased_prelu']:
            params_for_perf_key[f'{layer_act_name}_params'] = {} 
            prelu_num_params_choice = trial.suggest_categorical(f'{layer_act_name}_num_parameters_choice', ['global', 'per_channel'])
            if prelu_num_params_choice == 'global':
                params_for_perf_key[f'{layer_act_name}_params']['num_parameters'] = 1
            else: # 'per_channel'
                if layer_act_name == 'layer7_act2':
                    params_for_perf_key[f'{layer_act_name}_params']['num_parameters'] = 12
                else:
                    params_for_perf_key[f'{layer_act_name}_params']['num_parameters'] = 36
        
    # Batch size is for DataLoader, not directly for Model.__init__
    batch_size = 16
    
    # Other parameters that don't affect inference performance directly
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)

    # --- Performance Cache Lookup ---
    # Create a consistent, hashable key from the performance-relevant parameters
    # Sort the items to ensure the key is the same regardless of dictionary order
    cache_key_tuple = tuple(sorted(params_for_perf_key.items()))
    cache_key = json.dumps(cache_key_tuple) # Convert tuple to string for JSON dict key

    global performance_cache
    fps = None
    if cache_key in performance_cache:
        fps = performance_cache[cache_key]
        print(f"Trial {trial.number}: Found cached performance for this configuration: {fps:.2f} FPS. Skipping measurement.")
    else:    
        print(f"Trial {trial.number}: Performance not in cache. Measuring performance...")

    # Instantiate the model with the suggested hyperparameters
    model = Model( # Directly instantiate Model, passing all kwargs
        **params_for_perf_key, # Pass only model architecture parameters
        verbose=True
    )

    if model is None:
        print("Error: Model instantiation failed.")
        # If model instantiation fails, performance is effectively 0 or very bad
        fps = 0.0
        performance_cache[cache_key] = fps # Store failure in cache
        return float('inf')# , float('inf') # Return a very bad objective value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # Set model to evaluation mode for inference

    # Measure performance if not in cache (FPS in float32)
    if fps is None:
        fps = measure_performance(model, device)
        performance_cache[cache_key] = fps
        save_performance_cache()

    # --- Training Loop for Validation ---
    num_epochs = args.epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = model.perceptual_criterion # Use the model's defined criterion

    # Initialize GradScaler for mixed precision training (disabled for now)
    scaler = GradScaler('cuda', enabled=False)

    # --- Gather all available samples from the generator's train output directory ---
    expected_crop_size_tuple = tuple(args.crop_size)
    styles_set = set(args.styles_to_include) if args.styles_to_include is not None else None

    print(f"Gathering all available sample pairs from {args.generator_train_dir}...")
    all_available_samples = gather_all_samples_from_directory(
        directory_path=args.generator_train_dir,
        expected_crop_size=expected_crop_size_tuple,
        styles_to_include=styles_set,
        verbose=1
    )
    print(f"Found {len(all_available_samples)} total available sample pairs matching criteria.")

    # Check if any samples were found
    if not all_available_samples:
        print(f"Error: No sample pairs found in {args.generator_train_dir} matching the criteria. Check --generator_train_dir, --crop_size, and --styles_to_include.")
        sys.exit(1)

    # Calculate the number of samples for validation based on the ratio
    # Ensure val_split_ratio is between 0.0 and 1.0
    val_split_ratio = max(0.0, min(1.0, args.val_split_ratio))
    num_total_available = len(all_available_samples)
    num_val_available = int(num_total_available * val_split_ratio)
    num_train_available = num_total_available - num_val_available

    # Adjust split sizes to ensure at least one sample in each split if the total pool allows
    if num_total_available > 0:
        if num_train_available == 0:
            warnings.warn(f"Validation split ratio {val_split_ratio} results in 0 training samples. Adjusting to have 1 training sample.")
            num_train_available = 1
            num_val_available = num_total_available - 1
        if num_val_available == 0 and val_split_ratio > 0: # Only ensure val sample if ratio > 0 was requested
             warnings.warn(f"Validation split ratio {val_split_ratio} results in 0 validation samples. Adjusting to have 1 validation sample.")
             num_val_available = 1
             num_train_available = num_total_available - 1

    # Final check after potential adjustments
    if num_train_available <= 0: # Need at least one sample for training
        print("Error: Not enough samples available for the training split after applying validation ratio.")
        sys.exit(1)
    if num_val_available < 0: # Should not happen with previous logic, but defensive
         num_val_available = 0

    print(f"Splitting {num_total_available} available samples: {num_train_available} for train pool, {num_val_available} for validation pool.")

    # Split the list of sample pairs into train and validation pools
    train_pool_list = all_available_samples[:num_train_available]
    val_pool_list = all_available_samples[num_train_available:]

    # --- Create dataset instances using the split lists ---
    train_dataset = SRDataset(
        sample_pairs_list=train_pool_list,
        expected_crop_size=expected_crop_size_tuple,
        num_samples=args.train_samples
    )
    val_dataset = SRDataset(
        sample_pairs_list=val_pool_list,
        expected_crop_size=expected_crop_size_tuple,
        num_samples=args.val_samples
    )

    # Check if the resulting datasets' pools are empty
    if len(train_dataset.available_samples_pool) == 0:
        print("Error: Training dataset pool is empty after splitting. Cannot create DataLoader.")
        sys.exit(1)
    if len(val_dataset.available_samples_pool) == 0 and val_split_ratio > 0:
        print("Warning: Validation dataset pool is empty after splitting. Validation during training will use 0 samples.")

    num_dataloader_workers = os.cpu_count() // 2 or 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_dataloader_workers, pin_memory=True, persistent_workers=(num_dataloader_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_dataloader_workers, pin_memory=True, persistent_workers=(num_dataloader_workers > 0))

    num_batches = len(train_loader)

    print(f"Trial {trial.number}: Starting limited training ({num_epochs} epochs)...")
    try:
        start_time = time.time()
        best_val_loss = float("inf")
        for epoch in range(num_epochs):
            total_train_loss = 0.0
            model.train() # Set model to training mode
            optimizer.zero_grad() # Zero gradients for accumulation outside the batch loop
            for batch_idx, (lr_patches, hr_patches) in enumerate(train_loader):
                lr_patches = lr_patches.to(device, non_blocking=True)
                hr_patches = hr_patches.to(device, non_blocking=True)

                with autocast(device_type='cuda', enabled=False): # Explicitly specify device_type and enabled
                    outputs = model(lr_patches)
                    loss = criterion(outputs, hr_patches)

                # Scale loss for gradient accumulation
                loss = loss / args.accumulation_steps
                scaler.scale(loss).backward()
                
                # Perform optimization step only after accumulation_steps
                if (batch_idx + 1) % args.accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                if (batch_idx + 1) % (num_batches // 10) == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}] - {((batch_idx + 1) / num_batches) * 100:.2f}% batches processed")

                total_train_loss += loss.item() * args.accumulation_steps # Re-scale total_loss for correct average

                if total_train_loss == float('nan'):
                    print(f"Warning: NaN loss encountered during training in epoch {epoch}. Skipping further training for this trial.")
                    return float('inf')# , fps

            # After processing all batches, step the optimizer if accumulation is used
            # Handle remaining gradients if batches are not perfectly divisible
            if (len(train_loader) * batch_size) % args.accumulation_steps != 0:
                 scaler.step(optimizer)
                 scaler.update()
                 optimizer.zero_grad()

            avg_train_loss = total_train_loss / len(train_loader)

            print(f"Epoch [{epoch+1}/{num_epochs}] - 100% batches processed")
            print(f"Epoch [{epoch+1}/{num_epochs}] - Validating...")

            model.eval() # Set model in evaluation mode  
            val_loss = 0.0
            with torch.no_grad():
                for lr_batch, hr_batch in val_loader:
                    lr_batch = lr_batch.to(device)
                    hr_batch = hr_batch.to(device)
                    sr_batch = model(lr_batch)
                    batch_loss = model.criterion(sr_batch, hr_batch)
                    val_loss += batch_loss.item() * lr_batch.size(0)
                val_loss /= len(val_loader.dataset)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            trial.report(val_loss, epoch)

            print(f"Epoch [{epoch+1}/{num_epochs}] - Avg. train loss: {avg_train_loss:.4f}, Validation loss: {val_loss:.4f}")
            print(f"Epoch [{epoch+1}/{num_epochs}] - Done")

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        print(f"Trial {trial.number} done: Limited training finished. Final Avg Loss: {val_loss:.4f}")

        end_time = time.time()
        if end_time - start_time < 10:
            print(f"Warning: Trial {trial.number} took less than 10 seconds to complete. This may indicate an issue with the training loop or data loading.")
            return float('inf')# , fps
    
        return best_val_loss# , fps
    except optuna.exceptions.TrialPruned:
        # Re-raise the exception so Optuna can handle it
        raise
    except Exception as e:
        print(f"Error during training loop for trial {trial.number} with hyperparameters {trial.params}: {e}")
        return float('inf')# , float('inf') # Return infinity to indicate a failed trial

# --- Main Optimization Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optuna hyperparameter optimization for a 7-layer model.')
    parser.add_argument('--n_trials', type=int, required=True, help='Number of optimization trials.')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs per trial.')
    parser.add_argument('--study_name', type=str, required=True, help='Name of the Optuna study.')
    parser.add_argument('--storage', type=str, default='sqlite:///model_architecture_tuning.sqlite3', help='Optuna study storage URL.')
    parser.add_argument('--pruning_startup_trials', type=int, default=5, help='Number of startup trials before pruning starts.')
    parser.add_argument('--pruning_warmup_steps', type=int, default=5, help='Number of warmup steps before pruning starts.')
    parser.add_argument('--pruning_interval_steps', type=int, default=1, help='Number of steps after which to prune (0 for no pruning).')
    parser.add_argument('--generator_train_dir', type=str, required=True, help='Directory containing training images.')
    parser.add_argument('--train_samples', type=int, default=10000, help='Declared number of samples to use for training per epoch (epoch size).')
    parser.add_argument('--val_samples', type=int, default=1000, help='Declared number of samples to use for validation per epoch (epoch size).')
    parser.add_argument('--val_split_ratio', type=float, default=0.1, help='Ratio of the available sample pool to use for validation (0.0 to 1.0).')
    parser.add_argument("--styles_to_include", type=str, nargs='*', help="Optional list of specific style names (e.g., 'lores_rgb888_pNone_none') to include as inputs. If omitted, all styles are included.")
    parser.add_argument('--crop_size', type=int, nargs=2, required=True, help='Crop size (height, width) for patches. Used as patch_size[0] in SRDataset.')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps.')

    args = parser.parse_args()

    # Suppress specific UserWarning from Optuna regarding Trial.user_attrs
    warnings.filterwarnings("ignore", message="The `save_image` function is deprecated", category=UserWarning)

    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Load the performance cache at the start
    load_performance_cache()

    # Create or load a study
    pruner = MedianPruner(
        n_startup_trials=args.pruning_startup_trials,
        n_warmup_steps=args.pruning_warmup_steps,
        interval_steps=args.pruning_interval_steps,
    ) if args.pruning_interval_steps > 0 else optuna.pruners.NopPruner()

    print(f"Resuming existing study '{args.study_name}' or creating a new one.")
    study = optuna.create_study( # Changed from load_or_create_study
       # directions=["minimize", "maximize"],
        study_name=args.study_name,
        storage=args.storage,
        sampler=optuna.samplers.TPESampler(),
        pruner=pruner,
        load_if_exists=True
    )

    try:
        print(f"Starting optimization for {args.n_trials} trials.")
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")
    finally:
        # Save the performance cache when optimization finishes or is interrupted
        save_performance_cache()
    
    print("\nOptimization finished.")

    # Display results
    if study.best_trial:
        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        import optuna.visualization as vis
        if vis.is_available():
            try:
                print("\nGenerating Optuna visualizations...")
                os.makedirs("tuning_results", exist_ok=True)

                fig_history = vis.plot_optimization_history(study)
                fig_history.write_image(os.path.join("tuning_results", "optimization_history.png"))

                try:
                    fig_importance = vis.plot_param_importances(study)
                    fig_importance.write_image(os.path.join("tuning_results", "param_importances.png"))
                except Exception as e_imp:
                    print(f"Could not generate param_importances plot: {e_imp}")

                pareto = vis.plot_pareto_front(study)
                pareto.write_image(os.path.join("tuning_results", "pareto.png"))

                print("Optuna visualizations saved to the 'tuning_results' directory (if generated).")

            except Exception as e:
                print(f"Warning: Could not generate Optuna visualizations. Error: {e}")
        else:
            print("Plotly not available, skipping Optuna visualizations. Install with: pip install plotly")

    else:
        print("No trials completed successfully to determine the best trial, or study has no trials.")
