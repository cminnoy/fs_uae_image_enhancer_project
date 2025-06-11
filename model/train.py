import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
from torch.optim import lr_scheduler
from torchvision.utils import save_image
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchsummary import summary
from torchviz import make_dot
import argparse
import csv
import time
import sys
import glob
import random
from torch.amp import autocast, GradScaler
import shutil
import warnings
import numpy as np
from srdataset import SRDataset, gather_all_samples_from_directory
from gamma import srgb_to_linear_approx, linear_to_srgb_approx

import model_conv3
import model_conv5
import model_pix_shuffle

scaler = GradScaler(device='cuda')

def inference_on_directory(model, input_dir, output_dir, device):
    """
    Performs inference on RGB images in input_dir and saves model output to output_dir.
    Assumes:
    - Input: sRGB PNGs → converted to linear RGB [0, 1], upscaled to [0, 255]
    - Model input: float32 linear RGB [0, 255]
    - Model output: float32 linear RGB [0, 255]
    - Output saved as sRGB PNG
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    to_tensor = ToTensor()
    to_pil = ToPILImage(mode='RGB')

    model.eval()
    with torch.no_grad():
        for img_path in glob.glob(os.path.join(input_dir, "*.png")):
            try:
                img_pil = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}. Skipping.")
                continue

            # Convert to float32 tensor in [0,1]
            img_tensor = to_tensor(img_pil)  # (3, H, W)

            # Convert to linear RGB and downsample
            img_linear = srgb_to_linear_approx(img_tensor)# [:, ::2, ::2] # (3, H/2, W/2), linear RGB in [0,1]

            # Prepare input tensor
            input_tensor = img_linear.unsqueeze(0).to(device)

            # Inference
            output_tensor = model(input_tensor).squeeze(0).cpu() # (3, H, W), linear RGB in [0,1]

            # Convert back to sRGB
            output_srgb = linear_to_srgb_approx(output_tensor).clamp(0.0, 1.0)

            # Save image
            output_img = to_pil(output_srgb)
            output_path = os.path.join(output_dir, os.path.basename(img_path))
            output_img.save(output_path)
            print(f"Saved predicted image: {output_path}")

def save_training_stats(epoch, train_loss, val_loss, epochs_no_improve, learning_rate, checkpoint_path, csv_file='training_stats.csv'):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'EpochsNoImprove', 'LearningRate', 'Checkpoint Path'])
        writer.writerow([epoch, train_loss, val_loss, epochs_no_improve, learning_rate, checkpoint_path])

def load_last_epoch_and_checkpoint(lr, csv_file='training_stats.csv'):
    # Returns last epoch, best validation loss, best_epoch, epochs_no_improve, learning_rate, checkpoint path
    if not os.path.isfile(csv_file):
        return 0, float('inf'), 0, 0, lr, None

    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        rows = list(reader)
        if not rows:
            return 0, float('inf'), 0, 0, lr, None

        # Initialize variables to track the best validation loss
        best_val_loss = float('inf')
        best_epoch = 0
        last_epoch = int(rows[-1][0])
        epochs_no_improve = int(rows[-1][3])
        learning_rate = float(rows[-1][4])
        checkpoint_path = rows[-1][5]

        # Find the best validation loss and corresponding epoch
        for row in rows:
            val_loss = float(row[2])
            epoch = int(row[0])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch

        return last_epoch, best_val_loss, best_epoch, epochs_no_improve, learning_rate, checkpoint_path

# TensorBoard writer
writer = SummaryWriter(log_dir='runs/transformer_experiment')
average_inference_time = 0.0

def train_model(model, train_loader, val_loader,
                num_epochs=100,
                lr=0.1,
                checkpoint_interval=5,
                early_stopping_patience=10,
                device='cpu',
                accumulation_steps=16,
                checkpoint_dir='.',
                batch_size=1,
                inference_always=False):
    global average_inference_time
    model.to(device)

    # ----------------------------- 
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    stats_file = os.path.join(checkpoint_dir, f'training_stats_{args.model_type}.csv')
    start_epoch, best_val_loss, best_epoch, epochs_no_improve, lr, checkpoint_path = load_last_epoch_and_checkpoint(lr, stats_file)
    print(f"Starting training from epoch {start_epoch + 1} with best validation loss {best_val_loss:.4f}, epochs no improvement {epochs_no_improve}, learning rate {lr}")

    scaler = GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma = 0.955)

    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        ## TODO Move outside this function into main
        print(f"Loaded model from checkpoint: {checkpoint_path}")
    # ----------------------------- 

    for epoch in range(start_epoch + 1, num_epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        new_best = False

        optimizer.zero_grad(set_to_none=True)
        num_batches = len(train_loader)
        print(f"Epoch {epoch}/{num_epochs} - Training...")

        # Train
        for i, (lr_batch, hr_batch) in enumerate(train_loader):
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                # Forward pass
                sr_batch = model(lr_batch)
                batch_loss = model.criterion(sr_batch, hr_batch)

                # Scale and backpropagate the total loss
                scaled_loss = scaler.scale(batch_loss)
                scaled_loss.backward()
               # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                # Accumulate losses for reporting
                with torch.no_grad():                    
                    epoch_train_loss += batch_loss.item() * lr_batch.size(0)

            # Perform optimizer steps after scaling
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if (i + 1) % (num_batches // 10) == 0:
                print(f"Epoch [{epoch}/{num_epochs}] - {((i + 1) / num_batches) * 100:.2f}% batches processed")

        # Perform any remaining optimizer steps
        if (i + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        print(f"Epoch [{epoch}/{num_epochs}] - Done")
        time.sleep(5)

        # Average the losses
        epoch_train_loss = epoch_train_loss / len(train_loader.dataset)

        # Validation
        print(f"Validating...")
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for lr_batch, hr_batch in val_loader:
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)
                sr_batch = model(lr_batch)
                batch_loss = model.criterion(sr_batch, hr_batch)
                epoch_val_loss += batch_loss.item() * lr_batch.size(0)
            epoch_val_loss /= len(val_loader.dataset)

        # Step learning rate scheduler
        scheduler.step()
        time.sleep(5)

        # Update CSV and print
        writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/Validation', epoch_val_loss, epoch)
        if best_val_loss == 0 or not torch.isfinite(torch.tensor(best_val_loss)):
            difference_with_best = float('inf')
        else:
            difference_with_best = ((best_val_loss - epoch_val_loss) / best_val_loss) * 100
        print(f"Epoch [{epoch}/{num_epochs}]  Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}  Patience: {early_stopping_patience-epochs_no_improve} Difference with best: {difference_with_best:.4f}%, Learning Rate: {optimizer.param_groups[0]['lr']}")

        # Save best model checkpoint based on validation loss
        apply_inference = inference_always
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            apply_inference = True
            new_best = True
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_best_{args.model_type}.pth") 
            torch.save(model.state_dict(), checkpoint_path)
            full_model_path = os.path.join(checkpoint_dir, f"best_{args.model_type}.pt")
            torch.save(model, full_model_path)
            print('New best model saved.')
        else:
            epochs_no_improve += 1

        # Save checkpoint and sample outputs every checkpoint_interval epochs
        if epoch % checkpoint_interval == 0 or new_best or epochs_no_improve > early_stopping_patience or epoch == num_epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_{args.model_type}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

            # Save training statistics
            stats_file = os.path.join(checkpoint_dir, f'training_stats_{args.model_type}.csv')
            save_training_stats(epoch, epoch_train_loss, epoch_val_loss, epochs_no_improve, optimizer.param_groups[0]['lr'], checkpoint_path=checkpoint_path, csv_file=stats_file)

        if epochs_no_improve > early_stopping_patience:
            print("Early stopping triggered.")
            break

        # Perform inference on some amiga samples and save predicted results
        if apply_inference:
            original_dir = 'samples'
            predicted_dir = os.path.join(checkpoint_dir, 'predicted')

            # Measure inference time
            start_time = time.time()
            inference_on_directory(model, original_dir, predicted_dir, device)
            end_time = time.time()

            num_inference_images = len(glob.glob(os.path.join(original_dir, "*.png")))
            if num_inference_images > 0:
                inference_time = end_time - start_time
                average_inference_time = inference_time / num_inference_images
            else:
                 average_inference_time = 0.0
                 print(f"Warning: No images found for inference in {original_dir}.") 

            print(f"Inference completed for epoch {epoch}. Results saved to {predicted_dir}") # Line 212
            print(f"Average inference time per image: {average_inference_time:.4f} seconds") # Line 213

            # Save internals (optional; will be influenced by inference step)
            modules_to_check = []
            modules_to_check.append((model, 'basic'))

            # Iterate through the identified modules and save their internal images
            for module, prefix in modules_to_check:
                if hasattr(module, 'save') and isinstance(module.save, dict):
                    map = module.save
                    for key, item in map.items():
                        if item is not None and item.numel() > 0: # Check if tensor is not None and not empty
                            # Ensure item is a single image tensor before saving
                            if item.dim() == 4 and item.size(0) == 1:
                                image = item[0].cpu().clone().detach()
                                # Clamp values to [0, 1] before normalization and saving
                                image = torch.clamp(image, image.min(), image.max()) # Clamp before norm
                                # Normalize to [0, 1] for saving as image
                                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                                # Save with epoch_prefix_key format
                                save_filename = os.path.join(checkpoint_dir, f"epoch_{epoch}_{prefix}_{key}.png")
                                save_image(image, save_filename)
                            else:
                                 print(f"Warning: Skipping saving internal '{key}' from '{prefix}' as it's not a single image tensor (shape: {item.shape}).")
                        else:
                             print(f"Warning: Skipping saving internal '{key}' from '{prefix}' as it's None or empty.")

        time.sleep(10)

    return best_val_loss, best_epoch, average_inference_time

# Main Function: Prepare Data and Start Training
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an image enhancement model.')
    parser.add_argument('--model_type', type=str, required=True, choices=['conv3', 'conv3_heavy', 'conv5', 'conv5_heavy', 'pix_shuffle', 'pix_shuffle_heavy'],
                        help='Type of model to train: "conv3, conv3_heavy, conv5, conv5_heavy, conv6".')
    parser.add_argument('--edge_checkpoint_path', type=str, default=None,
                        help='Path to the trained checkpoint (.pth) to load for combined training.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--checkpoint_interval', type=int, default=5, help='Interval for saving checkpoints')
    parser.add_argument('--accumulation_steps', type=int, default=16, help='Gradient accumulation steps')
    parser.add_argument('--checkpoint_dir', type=str, default='.', help='Directory to load/store checkpoints')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--generator_train_dir', type=str, required=True, help='Output directory of the generator\'s train split (e.g., path/to/dataset_quantized/train).')
    parser.add_argument('--train_samples', type=int, default=10000, help='Declared number of samples to use for training per epoch (epoch size).')
    parser.add_argument('--val_samples', type=int, default=1000, help='Declared number of samples to use for validation per epoch (epoch size).')
    parser.add_argument('--val_split_ratio', type=float, default=0.1, help='Ratio of the available sample pool to use for validation (0.0 to 1.0).')
    parser.add_argument("--crop_size", type=int, nargs=2, default=[752, 576], help="Expected crop size as W H (e.g., 752 576) for images in the dataset. Defaults to 752x576.")
    parser.add_argument("--styles_to_include", type=str, nargs='*', help="Optional list of specific style names (e.g., 'lores_rgb888_pNone_none') to include as inputs. If omitted, all styles are included.")
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level for warnings/messages (0: no warnings, 1: basic, 2: detailed).')
    parser.add_argument('--inference_always', action='store_true', help='Run inference on the Amiga sample directory after training, regardless of training improvement.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Start learning rate (default 0.001)')

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Selected model type for training: {args.model_type}")
    if args.model_type == 'conv3':
        model = model_conv3.get_model('lightweight')
        print("Using Conv2D model with 3 layers; lightweight.")
    elif args.model_type == 'conv3_heavy':
        model = model_conv3.get_model('heavyweight')
        print("Using Conv2D model with 3 layers; heavyweight.")
    elif args.model_type == "conv5":
        model = model_conv5.get_model('lightweight')
        print("Using Conv2D model with 5 layers; lightweight.")
    elif args.model_type == "conv5_heavy":
        model = model_conv5.get_model('heavyweight')
        print("Using Conv2D model with 5 layers; heavyweight.")
    elif args.model_type == "pix_shuffle":
        model = model_pix_shuffle.get_model('lightweight')
        print("Based on CRN and ESPCN; lightweight.")
    elif args.model_type == "pix_shuffle_heavy":
        model = model_pix_shuffle.get_model('heavyweight')
        print("Based on CRN and ESPCN; heavyweight.")
    else:
        print(f"Error: Unknown model type '{args.model_type}'.")
        sys.exit(1)
    
    # Move model to GPU
    model = model.to(device)

    # --- Gather all available samples from the generator's train output directory ---
    expected_crop_size_tuple = tuple(args.crop_size)
    styles_set = set(args.styles_to_include) if args.styles_to_include is not None else None

    print(f"Gathering all available sample pairs from {args.generator_train_dir}...")
    all_available_samples = gather_all_samples_from_directory(
        directory_path=args.generator_train_dir,
        expected_crop_size=expected_crop_size_tuple,
        styles_to_include=styles_set,
        verbose=args.verbose
    )
    print(f"Found {len(all_available_samples)} total available sample pairs matching criteria.")

    # Check if any samples were found
    if not all_available_samples:
        print(f"Error: No sample pairs found in {args.generator_train_dir} matching the criteria. Check --generator_train_dir, --crop_size, and --styles_to_include.")
        sys.exit(1)

    # --- Perform Train/Validation Split Programmatically ---
    # Shuffle the list of all available samples before splitting
    random.shuffle(all_available_samples)

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


    # Split the shuffled list of sample pairs into train and validation pools
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


    # Create data loaders using the dataset objects
    # num_workers depends on available CPU cores and system
    # persistent_workers=True is good practice with num_workers > 0
    # Pin_memory=True can speed up data transfer to GPU
    num_dataloader_workers = os.cpu_count() // 2 or 0
    if num_dataloader_workers > 0:
         print(f"Using {num_dataloader_workers} workers for DataLoaders.")
    else:
         print("Using main process for DataLoaders (num_workers=0).")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_dataloader_workers, pin_memory=True, persistent_workers=(num_dataloader_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_dataloader_workers, pin_memory=True, persistent_workers=(num_dataloader_workers > 0))

    # Start training (assuming train_model function is defined elsewhere and accepts these loaders)
    best_val, best_epoch, average_inference_time = train_model(
                          model=model,
                          train_loader=train_loader,
                          val_loader=val_loader,
                          num_epochs=args.epochs,
                          lr=args.learning_rate,
                          checkpoint_interval=args.checkpoint_interval,
                          early_stopping_patience=args.early_stopping_patience,
                          device=device,
                          accumulation_steps=args.accumulation_steps,
                          checkpoint_dir=args.checkpoint_dir,
                          batch_size=args.batch_size,
                          inference_always=args.inference_always)

    print(f"Average inference time: {average_inference_time:.4f} seconds")
    print(f"Best validation loss: {best_val:.4f} at epoch {best_epoch}")
    writer.close()
    sys.exit(0)
