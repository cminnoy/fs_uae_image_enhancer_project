import os, time, argparse, warnings, sys
from typing import Tuple
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# Removed: from torch.amp import autocast, GradScaler

import optuna
from optuna.pruners import MedianPruner
from model_conv6 import Model # Ensure Model class is imported for instantiation in caching
from srdataset import SRDataset, gather_all_samples_from_directory

# Assuming SSIMLoss is available from pytorch_msssim or a custom loss file
# If not, you might need to install it: pip install pytorch-msssim
# Or import from your local loss_vgg.py if SSIMLoss is defined there.
try:
    from pytorch_msssim import SSIM as SSIMLoss
except ImportError:
    warnings.warn("pytorch_msssim not found. SSIMLoss will be a dummy function returning 0.0.")
    class SSIMLoss(nn.Module):
        def forward(self, x, y):
            return torch.tensor(0.0, device=x.device)

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

# --- Helper function for FPS measurement ---
# Extracted the FPS measurement logic into a separate function
def _measure_model_fps(model: nn.Module, device: torch.device) -> float:
    """
    Measures the inference frames per second (FPS) of a given model.
    """
    model.eval() # Set model to evaluation mode

    # Dummy input based on common Super-Resolution input size (e.g., 576x752 for 3 channels)
    # This matches the input example used in the original model_conv6.py snippet.
    # Using float32 for consistency with full precision training now.
    x = torch.rand((1, 3, 576, 752), dtype=torch.float32).to(device)

    # Warm-up phase
    with torch.no_grad():
        for _ in range(20): # Run 20 warm-up iterations
            _ = model(x)

    # Measure FPS over a fixed duration (e.g., 20 seconds)
    start_time = time.time()
    num_iterations = 0

    with torch.no_grad():
        while time.time() - start_time < 20: # Measure for 20 seconds
            _ = model(x)
            num_iterations += 1

    elapsed_time = time.time() - start_time
    fps = num_iterations / elapsed_time if elapsed_time > 0 else 0.0
    return fps

# --- R2 Objective Function ---
def objective(
    trial: optuna.Trial,
    epochs: int, # Now passed as an argument
    train_data_dir: str,
    train_samples_limit: int, # Renamed to avoid conflict with `train_samples` in SRDataset logic if any
    val_samples_limit: int, # Renamed
    val_split_ratio: float,
    crop_size: Tuple[int, int]
) -> float:
    # Set the batch size to 16 as requested
    batch_size = 16

    # --- Hyperparameters for the Model Architecture ---
    # These parameters determine the structure of the model.
    # As per user's request, layer_out_channels and layer_kernel_size
    # are NOT searched by Optuna in this study, except for layer6_kernel_size.
    # The Model class will use its default values for other parameters.

    # Layer 6 (Output Layer - channels fixed to 3 for RGB output)
    layer6_kernel_size = trial.suggest_int("layer6_kernel_size", 3, 7, step=2)

    # Default channel sizes for PReLU `num_parameters` determination and model config key
    # These match the default values in model_conv6.py's Model.__init__
    default_layer_channels = {
        'layer1': 36,
        'layer2': 36,
        'layer3': 36,
        'layer4': 36,
        'layer5': 36,
        'layer6': 3 # Output layer channels (fixed)
    }

    # Default kernel sizes for model config key (Model uses 3x3 for all by default except layer6_kernel_size)
    default_layer_kernel_sizes = {
        'layer1_kernel_size': 3,
        'layer2_kernel_size': 3,
        'layer3_kernel_size': 3,
        'layer4_kernel_size': 3,
        'layer5_kernel_size': 3,
        # layer6_kernel_size is chosen by Optuna
    }

    # --- Activation Functions Choices ---
    # UPDATED: Synchronized with the list provided by your `activations.py` error message.
    activation_choices = [
        'identity', 'elu', 'gelu', 'leaky_relu', 'mish', 'prelu',
        'relu', 'relu6', 'sigmoid', 'silu', 'swish', 'softplus',
        'tanh', 'log_softmax', 'softmax', 'scaled_tanh', 'telu',
        'sinlu', 'biased_relu', 'biased_prelu'
    ]

    # Helper function to suggest parameters for activation functions
    def _get_activation_params(trial_obj: optuna.Trial, prefix: str, act_name: str, current_out_channels: int) -> dict:
        params = {}
        if act_name == 'leaky_relu':
            # Negative slope for LeakyReLU, typically a small positive number
            params['negative_slope'] = trial_obj.suggest_float(f'{prefix}_negative_slope', 0.001, 0.5, log=True)
        elif act_name == 'prelu': # ONLY for PReLU now, as BiasedPReLU does not accept 'num_parameters'
            # For PReLU, num_parameters can be 1 (shared) or equal to the number of channels
            prelu_num_params_choice = trial_obj.suggest_categorical(f'{prefix}_num_parameters_choice', ['global', 'per_channel'])
            if prelu_num_params_choice == 'global':
                params['num_parameters'] = 1
            else: # 'per_channel'
                params['num_parameters'] = current_out_channels
            # The 'init' value (initial weight for negative slope) for PReLU is usually 0.25,
            # we'll keep it default unless you want to search for it explicitly.
        elif act_name == 'elu':
            # Alpha parameter for ELU, typically positive
            params['alpha'] = trial_obj.suggest_float(f'{prefix}_alpha', 0.1, 2.0)
        elif act_name == 'gelu':
            # Approximate parameter for GELU, 'none' for exact, 'tanh' for tanh approximation
            params['approximate'] = trial_obj.suggest_categorical(f'{prefix}_approximate', ['none', 'tanh'])
        elif act_name == 'log_softmax' or act_name == 'softmax':
            # Explicitly set dim for log_softmax and softmax to avoid UserWarning
            params['dim'] = 1 # Assuming channel dimension for image processing
        # Add other activation-specific parameters here if needed
        return params

    # Layer 1 Activations
    layer1_act1 = trial.suggest_categorical("layer1_act1", activation_choices)
    layer1_act1_params = _get_activation_params(trial, "layer1_act1", layer1_act1, default_layer_channels['layer1'])

    layer1_act2 = trial.suggest_categorical("layer1_act2", activation_choices)
    layer1_act2_params = _get_activation_params(trial, "layer1_act2", layer1_act2, default_layer_channels['layer1'])

    # Layer 2 Activations
    layer2_act1 = trial.suggest_categorical("layer2_act1", activation_choices)
    layer2_act1_params = _get_activation_params(trial, "layer2_act1", layer2_act1, default_layer_channels['layer2'])

    layer2_act2 = trial.suggest_categorical("layer2_act2", activation_choices)
    layer2_act2_params = _get_activation_params(trial, "layer2_act2", layer2_act2, default_layer_channels['layer2'])

    layer2_act3 = trial.suggest_categorical("layer2_act3", activation_choices)
    layer2_act3_params = _get_activation_params(trial, "layer2_act3", layer2_act3, default_layer_channels['layer2'])

    layer2_act4 = trial.suggest_categorical("layer2_act4", activation_choices)
    layer2_act4_params = _get_activation_params(trial, "layer2_act4", layer2_act4, default_layer_channels['layer2'])

    # Layer 3 Activations
    layer3_act1 = trial.suggest_categorical("layer3_act1", activation_choices)
    layer3_act1_params = _get_activation_params(trial, "layer3_act1", layer3_act1, default_layer_channels['layer3'])

    layer3_act2 = trial.suggest_categorical("layer3_act2", activation_choices)
    layer3_act2_params = _get_activation_params(trial, "layer3_act2", layer3_act2, default_layer_channels['layer3'])

    # Layer 4 Activations
    layer4_act1 = trial.suggest_categorical("layer4_act1", activation_choices)
    layer4_act1_params = _get_activation_params(trial, "layer4_act1", layer4_act1, default_layer_channels['layer4'])

    layer4_act2 = trial.suggest_categorical("layer4_act2", activation_choices)
    layer4_act2_params = _get_activation_params(trial, "layer4_act2", layer4_act2, default_layer_channels['layer4'])

    # Layer 5 Activations
    layer5_act1 = trial.suggest_categorical("layer5_act1", activation_choices)
    layer5_act1_params = _get_activation_params(trial, "layer5_act1", layer5_act1, default_layer_channels['layer5'])

    layer5_act2 = trial.suggest_categorical("layer5_act2", activation_choices)
    layer5_act2_params = _get_activation_params(trial, "layer5_act2", layer5_act2, default_layer_channels['layer5'])

    # Layer 6 Activations (Output Layer)
    layer6_act1 = trial.suggest_categorical("layer6_act1", activation_choices)
    layer6_act1_params = _get_activation_params(trial, "layer6_act1", layer6_act1, default_layer_channels['layer6']) # Output is 3 channels

    layer6_act2 = trial.suggest_categorical("layer6_act2", activation_choices)
    layer6_act2_params = _get_activation_params(trial, "layer6_act2", layer6_act2, default_layer_channels['layer6']) # Output is 3 channels

    # --- Learning Rate ---
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)

    # Construct the model configuration dictionary to use as a cache key
    # This replaces the need for model.get_config()
    model_config = {
        'layer1_out_channels': default_layer_channels['layer1'],
        'layer1_kernel_size': default_layer_kernel_sizes['layer1_kernel_size'],
        'layer1_act1': layer1_act1,
        'layer1_act1_params': layer1_act1_params,
        'layer1_act2': layer1_act2,
        'layer1_act2_params': layer1_act2_params,

        'layer2_out_channels': default_layer_channels['layer2'],
        'layer2_kernel_size': default_layer_kernel_sizes['layer2_kernel_size'],
        'layer2_act1': layer2_act1,
        'layer2_act1_params': layer2_act1_params,
        'layer2_act2': layer2_act2,
        'layer2_act2_params': layer2_act2_params,
        'layer2_act3': layer2_act3,
        'layer2_act3_params': layer2_act3_params,
        'layer2_act4': layer2_act4,
        'layer2_act4_params': layer2_act4_params,

        'layer3_out_channels': default_layer_channels['layer3'],
        'layer3_kernel_size': default_layer_kernel_sizes['layer3_kernel_size'],
        'layer3_act1': layer3_act1,
        'layer3_act1_params': layer3_act1_params,
        'layer3_act2': layer3_act2,
        'layer3_act2_params': layer3_act2_params,

        'layer4_out_channels': default_layer_channels['layer4'],
        'layer4_kernel_size': default_layer_kernel_sizes['layer4_kernel_size'],
        'layer4_act1': layer4_act1,
        'layer4_act1_params': layer4_act1_params,
        'layer4_act2': layer4_act2,
        'layer4_act2_params': layer4_act2_params,

        'layer5_out_channels': default_layer_channels['layer5'],
        'layer5_kernel_size': default_layer_kernel_sizes['layer5_kernel_size'],
        'layer5_act1': layer5_act1,
        'layer5_act1_params': layer5_act1_params,
        'layer5_act2': layer5_act2,
        'layer5_act2_params': layer5_act2_params,

        'layer6_out_channels': default_layer_channels['layer6'],
        'layer6_kernel_size': layer6_kernel_size, # This one is optimized
        'layer6_act1': layer6_act1,
        'layer6_act1_params': layer6_act1_params,
        'layer6_act2': layer6_act2,
        'layer6_act2_params': layer6_act2_params,
    }

    model_config_key = json.dumps(model_config, sort_keys=True)

    # Check if this configuration has been measured before
    if model_config_key in performance_cache:
        # If found, return the cached value directly
        fps_cached = performance_cache[model_config_key]
        print(f"Using cached FPS for model config: {fps_cached:.2f} FPS")
        return 0.0 # Return a placeholder value; the true objective is the training loss

    # Instantiate the model with suggested hyperparameters
    model = Model(
        layer6_kernel_size=layer6_kernel_size, # Only layer6_kernel_size is searched
        layer1_act1=layer1_act1,
        layer1_act1_params=layer1_act1_params,
        layer1_act2=layer1_act2,
        layer1_act2_params=layer1_act2_params,

        layer2_act1=layer2_act1,
        layer2_act1_params=layer2_act1_params,
        layer2_act2=layer2_act2,
        layer2_act2_params=layer2_act2_params,
        layer2_act3=layer2_act3,
        layer2_act3_params=layer2_act3_params,
        layer2_act4=layer2_act4,
        layer2_act4_params=layer2_act4_params,

        layer3_act1=layer3_act1,
        layer3_act1_params=layer3_act1_params,
        layer3_act2=layer3_act2,
        layer3_act2_params=layer3_act2_params,

        layer4_act1=layer4_act1,
        layer4_act1_params=layer4_act1_params,
        layer4_act2=layer4_act2,
        layer4_act2_params=layer4_act2_params,

        layer5_act1=layer5_act1,
        layer5_act1_params=layer5_act1_params,
        layer5_act2=layer5_act2,
        layer5_act2_params=layer5_act2_params,

        layer6_act1=layer6_act1,
        layer6_act1_params=layer6_act1_params,
        layer6_act2=layer6_act2,
        layer6_act2_params=layer6_act2_params,
    )

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss function and optimizer
    criterion_loss = torch.nn.L1Loss()
    criterion_ssim = SSIMLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # DataLoader setup using passed arguments
    try:
        if not os.path.exists(train_data_dir):
            raise FileNotFoundError(f"Training samples path not found: {train_data_dir}")

        print(f"Gathering training samples from: {train_data_dir}")
        # Pass expected_crop_size to gather_all_samples_from_directory
        all_image_files = gather_all_samples_from_directory(train_data_dir, expected_crop_size=crop_size)

        if not all_image_files:
            print(f"No image files found in {train_data_dir}. Please check your dataset.")
            return float('inf')

        # Limit training samples if specified
        if len(all_image_files) > train_samples_limit:
            image_files = all_image_files[:train_samples_limit]
            print(f"Limited training image files to {len(image_files)}.")
        else:
            image_files = all_image_files

        # Note: val_samples_limit and val_split_ratio are passed but not actively used
        # for splitting the dataset here. SRDataset or a separate validation loop
        # would need to implement this. crop_size is also passed but not directly
        # used in SRDataset constructor, assuming SRDataset handles cropping internally.

        # Pass expected_crop_size and num_samples to SRDataset constructor
        dataset = SRDataset(image_files, expected_crop_size=crop_size, num_samples=train_samples_limit)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_ssim = 0
        num_batches = 0

        for batch_idx, (low_res, high_res) in enumerate(dataloader):
            low_res, high_res = low_res.to(device), high_res.to(device)

            optimizer.zero_grad()

            output = model(low_res)
            loss = criterion_loss(output, high_res)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            with torch.no_grad():
                total_ssim += criterion_ssim(output, high_res).item()

            trial.report(loss.item(), epoch * len(dataloader) + batch_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        avg_loss = total_loss / num_batches
        avg_ssim = total_ssim / num_batches if num_batches > 0 else 0

        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, Avg SSIM: {avg_ssim:.4f}")

    try:
        # Call the new helper function to measure FPS
        fps_measured = _measure_model_fps(model, device)
        performance_cache[model_config_key] = fps_measured
        save_performance_cache()
        print(f"Measured FPS for current model config: {fps_measured:.2f} FPS")

    except Exception as e:
        print(f"Error measuring FPS: {e}")
        return float('inf')

    return avg_loss

# --- Main execution block for Optuna study ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna study for pruning activation functions.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs per trial.")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna trials.")
    parser.add_argument("--study_name", type=str, default="prune_act_r2", help="Name of the Optuna study.")
    parser.add_argument("--generator_train_dir", type=str, required=True,
                        help="Path to the directory containing training samples (low-res/high-res pairs).")
    parser.add_argument("--train_samples", type=int, default=10000,
                        help="Number of training samples to use from the dataset.")
    parser.add_argument("--val_samples", type=int, default=1000,
                        help="Number of validation samples to use.")
    parser.add_argument("--val_split_ratio", type=float, default=0.1,
                        help="Ratio of validation samples to total samples if splitting from train_samples.")
    parser.add_argument("--crop_size", type=int, nargs=2, default=[376, 288],
                        help="Height and width for cropping input images (e.g., --crop_size 376 288).")

    args = parser.parse_args()

    # Load performance cache at the start of the script
    load_performance_cache()

    # Create Optuna study
    # Using MedianPruner as an example; you can adjust its parameters.
    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{args.study_name}.db", # Persist study results to a SQLite database
        direction="minimize", # We are minimizing L1 loss
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10),
        load_if_exists=True # Continue study if it already exists
    )

    print(f"Starting Optuna study '{args.study_name}' with {args.n_trials} trials...")
    print(f"Training data directory: {args.generator_train_dir}")
    print(f"Epochs per trial: {args.epochs}")

    # Pass relevant arguments to the objective function using a lambda
    study.optimize(
        lambda trial: objective(
            trial,
            epochs=args.epochs,
            train_data_dir=args.generator_train_dir,
            train_samples_limit=args.train_samples,
            val_samples_limit=args.val_samples,
            val_split_ratio=args.val_split_ratio,
            crop_size=tuple(args.crop_size)
        ),
        n_trials=args.n_trials
    )

    # --- Optuna Result Visualization ---
    if study.best_trial:
        print("\nBest trial found:")
        trial = study.best_trial
        print(f"  Value (minimized L1 loss): {trial.value:.6f}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        # Optuna visualization plots
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

                # Note: plot_pareto_front is typically used for multi-objective optimization.
                # Since we are minimizing a single objective (L1 loss), it might not be as relevant,
                # but I'll keep it if you intend to add another objective later (e.g., FPS).
                # For a single objective, plot_optimization_history and plot_param_importances are key.
                # pareto = vis.plot_pareto_front(study)
                # pareto.write_image(os.path.join("tuning_results", "pareto.png"))

                print("Optuna visualizations saved to the 'tuning_results' directory.")

            except Exception as e:
                print(f"Warning: Could not generate Optuna visualizations. Error: {e}")
        else:
            print("Plotly not available, skipping Optuna visualizations. Install with: pip install plotly")

    else:
        print("No trials completed successfully to determine the best trial, or study has no trials.")