import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import argparse
import os
import sys
import time
import shutil
import random

import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from PIL import Image
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# --------------------------------------------------------------------------------
# Path Setup
# --------------------------------------------------------------------------------
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path)) 
if project_root not in sys.path:
    sys.path.append(project_root)
from model_residual_unet import get_model
from srdataset import SRDataset, gather_all_samples_from_directory, add_size_argument 
from loss_vgg import PerceptualLoss
import gamma

# ------------------------------------------------------------
# Early stopping helper (Enhanced for Serialization)
# ------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        is_best = False

        if self.best_loss is None:
            self.best_loss = val_loss
            is_best = True
            return False, True

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            is_best = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop, is_best

    def state_dict(self):
        return {
            'best_loss': self.best_loss,
            'counter': self.counter,
            'should_stop': self.should_stop
        }

    def load_state_dict(self, state_dict):
        self.best_loss = state_dict.get('best_loss')
        self.counter = state_dict.get('counter', 0)
        self.should_stop = state_dict.get('should_stop', False)

# ------------------------------------------------------------
# Sample Visualizer for TensorBoard
# ------------------------------------------------------------
class Visualizer:
    def __init__(self, sample_dir, device, writer):
        self.device = device
        self.writer = writer
        self.sample_tensors = []

        if os.path.exists(sample_dir):
            files = sorted([f for f in os.listdir(sample_dir) if f.endswith('.png')])
            for f in files:
                img = Image.open(os.path.join(sample_dir, f)).convert('RGB')
                t = TF.to_tensor(img).unsqueeze(0).to(self.device)
                self.sample_tensors.append(t)

    def log_epoch(self, model, epoch):
        model.eval()
        comparisons = []
        with torch.no_grad():
            for img_t in self.sample_tensors:
                output = model(img_t)

                # Sanitize for TensorBoard visualization
                output = torch.nan_to_num(output, nan=0.0).clamp(0, 1)

                combined = torch.cat([img_t, output], dim=3)
                comparisons.append(combined.squeeze(0).cpu())

        grid = make_grid(comparisons, nrow=1)
        self.writer.add_image('Visual_Progress/Samples', grid, epoch)
        model.train()

# --------------------------------------------------------------------------------
# Trainer Class Definition
# --------------------------------------------------------------------------------
class Trainer:
    """
    Handles the DDP training and checkpointing logic for the ResidualUNet model.
    """
    def __init__(self, args, local_rank):
        self.args = args
        self.local_rank = local_rank

        # Set device based on local rank
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")

        # Only rank 0 handles logging and printing
        self.is_master = (self.local_rank == 0)
        if self.is_master:
            self.writer = SummaryWriter(log_dir=args.log_dir)

        self.scaler = GradScaler()
        self.current_epoch = 0
        self.start_time = time.time()

        # Initialize Early Stopping
        self.early_stopper = EarlyStopping(
            patience=args.early_stop_patience, 
            min_delta=args.early_stop_delta
        )

        self.setup_data()
        self.setup_model_and_optimizer()

    def setup_data(self):
        # 1. Gather all unique file pairs from all provided directories
        all_pairs = []

        for d in self.args.data_dir:
            if self.is_master:
                try:
                    print(f"Gathering samples from: {d}")
                except Exception as e:
                    print(f"Error occurred while gathering samples from {d}: {e}")

            if d:
                try:
                    dir_pairs = gather_all_samples_from_directory(
                        directory_path=d,
                        expected_crop_size=self.args.generator_crop_size,
                        styles_to_include=None,
                        verbose=2 if self.is_master else 0
                    )
                    all_pairs.extend(dir_pairs)
                except Exception as e:
                    print(f"Error occurred while gathering samples from {d}: {e}")
                    raise e

        if self.is_master:
            print(f"Total aggregated pairs from {len(self.args.data_dir)} directories: {len(all_pairs)}")

        # 2. Deterministic Shuffle and Physical Split
        # We seed here so every DDP rank performs the EXACT same split
        random.seed(42)
        random.shuffle(all_pairs)

        val_count = int(len(all_pairs) * self.args.val_fraction)
        val_pairs = all_pairs[:val_count]
        train_pairs = all_pairs[val_count:]

        # 3. Handle training crop size tuple conversion
        if isinstance(self.args.train_crop_size, int):
            crop_size = (self.args.train_crop_size, self.args.train_crop_size)
        else:
            crop_size = self.args.train_crop_size

        # 4. Initialize separate Dataset instances
        # Train dataset uses your virtual epoch length (samples_per_epoch)
        self.train_dataset = SRDataset(
            sample_pairs_list=train_pairs,
            generator_output_crop_size=self.args.generator_crop_size,
            train_crop_size=crop_size
        )

        # Validation dataset uses all available validation samples.
        # A rotating validation window limits how many are checked per epoch.
        self.val_dataset = SRDataset(
            sample_pairs_list=val_pairs, 
            generator_output_crop_size=self.args.generator_crop_size, 
            train_crop_size=crop_size 
        )

        # Rotating validation window: start offset and per-epoch max samples (configurable)
        self.val_limit = min(int(self.args.val_limit), len(val_pairs)) if len(val_pairs) > 0 else 0
        self.val_offset = 0

        if self.is_master:
            print(f"Total pairs: {len(all_pairs)} | Train pool: {len(train_pairs)} | Val pool: {len(val_pairs)}")
            print(f"Epoch configuration -> Train steps: {self.args.samples_per_epoch} | Val steps: {self.args.val_limit} (rotating window)")

        # Train Distributed Sampler
        self.train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=self.args.shuffle_data,
            drop_last=True
        )

        # Train DataLoader
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=False, # Sampler handles shuffling
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=self.train_sampler
        )

    def setup_model_and_optimizer(self):
        if self.is_master:
            print(f"Initializing model: {self.args.model_type}")
        self.model = get_model(self.args.model_type, lores_only=self.args.lores_only, verbose=self.args.verbose).to(self.device)
        if self.is_master and self.args.print_model_layers:
            print(self.model)

        world_size = dist.get_world_size()
        total_batches_per_epoch = self.args.samples_per_epoch // (self.args.batch_size * world_size)
        self.learning_rate = self.args.learning_rate
        self.steps_per_epoch = total_batches_per_epoch

        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, # The scheduler will override this
            betas=(self.args.adam_beta1, self.args.adam_beta2)
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=self.steps_per_epoch * 5,
            T_mult=2,
            eta_min=1e-7
        )

        # Load checkpoint if specified
        if self.args.load_checkpoint:
            self.load_checkpoint(self.args.load_checkpoint)

        # Wrap the model with DDP
        self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=self.args.find_unused_parameters)

    @torch.inference_mode()
    def validate(self):
        """
        Runs validation loop and returns global average validation loss.
        """
        self.model.eval()

        # If there are no validation samples, return 0.0
        val_dataset_len = len(self.val_dataset)
        if val_dataset_len == 0 or self.val_limit == 0:
            self.model.train()
            return 0.0

        # Build the rotating slice of indices for this validation run
        start = int(self.val_offset) % val_dataset_len
        end = start + int(self.val_limit)
        if end <= val_dataset_len:
            indices = list(range(start, end))
        else:
            # Wrap around
            indices = list(range(start, val_dataset_len)) + list(range(0, end - val_dataset_len))

        # Create a Subset and DataLoader for this validation slice, using DistributedSampler
        from torch.utils.data import Subset
        subset = Subset(self.val_dataset, indices)
        subset_sampler = DistributedSampler(
            subset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
            drop_last=False
        )

        val_batch_size = 12
        val_num_workers = self.args.num_workers
        val_pin_memory = False

        val_loader = DataLoader(
            subset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=val_pin_memory,
            drop_last=False,
            sampler=subset_sampler,
            prefetch_factor=2
        )

        total_val_loss = 0.0
        num_batches = 0
        local_samples_processed = 0
        log_interval = getattr(self.args, 'val_log_interval', 50)

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with autocast(device_type=self.device.type, dtype=dtype, enabled=self.args.use_amp):
            for batch in val_loader:
                if isinstance(batch, dict):
                    lr = batch["lr"].to(self.device)
                    hr = batch["hr"].to(self.device)
                else:
                    lr, hr = batch
                    lr = lr.to(self.device)
                    hr = hr.to(self.device)

                out = self.model(lr)
                loss = self.model.module.criterion(out, hr)
                total_val_loss += loss.item()
                num_batches += 1
                # Track processed samples for progress logging
                batch_n = hr.size(0) if hasattr(hr, 'size') else 1
                local_samples_processed += int(batch_n)

                # Periodically aggregate progress across ranks and log from master
                if (num_batches % log_interval) == 0 or local_samples_processed >= len(indices):
                    if dist.is_initialized():
                        proc_tensor = torch.tensor([local_samples_processed], dtype=torch.long, device=self.device)
                        dist.all_reduce(proc_tensor, op=dist.ReduceOp.SUM)
                        global_processed = int(proc_tensor.item())
                    else:
                        global_processed = local_samples_processed

                    if self.is_master:
                        print(f"    Validation progress: {global_processed}/{len(indices)} samples ({(global_processed/ max(1,len(indices)))*100:.1f}%)")

            # 1. Calculate local average
            local_avg_loss = total_val_loss / max(1, num_batches)

            # 2. Aggregate across all GPUs
            loss_tensor = torch.tensor([local_avg_loss], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            global_val_loss = loss_tensor.item() / dist.get_world_size()

            # Advance the rotating offset for next epoch (only local update; persisted in checkpoints)
            self.val_offset = (self.val_offset + self.val_limit) % val_dataset_len

        self.model.train()
        return global_val_loss

    def save_checkpoint(self, is_best: bool, epoch: int, loss: float, val_loss: float):
        if not self.is_master:
            return

        os.makedirs(self.args.checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'val_loss': val_loss,
            'early_stopping_state': self.early_stopper.state_dict(),
            'val_offset': getattr(self, 'val_offset', 0),
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'args': self.args.__dict__,
            'time_elapsed': time.time() - self.start_time
        }

        filepath = os.path.join(self.args.checkpoint_dir, f'epoch_{epoch:04d}.pth')
        torch.save(checkpoint, filepath)

        if is_best:
            best_filepath = os.path.join(self.args.checkpoint_dir, 'best_model.pth')
            shutil.copyfile(filepath, best_filepath)
            print(f"Saved best model (Val Loss: {val_loss:.6f}) to {best_filepath}")

        print(f"Saved checkpoint to {filepath}")

    def load_checkpoint(self, checkpoint_path: str):
        """Loads a checkpoint and resumes training."""
        if not os.path.exists(checkpoint_path):
            if self.is_master:
                print(f"Warning: Checkpoint file not found at {checkpoint_path}")
            return

        if self.is_master:
            print(f"Loading checkpoint from {checkpoint_path}...")
        try:
            # Note: Load model onto CPU first if loading happens before DDP wrapping
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Use the model's load_state_dict *before* DDP wrapping
            self.model.load_state_dict(checkpoint['model_state_dict'])

            if not self.args.reset_optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                self.current_epoch = checkpoint['epoch'] + 1
                self.start_time -= checkpoint.get('time_elapsed', 0.0)
                if 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                # Restore Early Stopping State
                if 'early_stopping_state' in checkpoint:
                    self.early_stopper.load_state_dict(checkpoint['early_stopping_state'])
                    self.early_stopper.patience = self.args.early_stop_patience
                    self.early_stopper.min_delta = self.args.early_stop_delta
                    self.early_stopper.should_stop = False
                    # Restore validation offset if present
                    self.val_offset = checkpoint.get('val_offset', getattr(self, 'val_offset', 0))
                else:
                    # Backward compatibility for older checkpoints
                    old_best = checkpoint.get('best_loss')
                    if old_best: self.early_stopper.best_loss = old_best

                if self.is_master:
                    print(f"Resuming from epoch {self.current_epoch}.")
            else:
                if self.is_master:
                    print("Loaded model weights, but reset optimizer and epoch count.")

        except Exception as e:
            if self.is_master:
                print(f"Error loading checkpoint: {e}")
                print("Starting training from scratch.")

    def train_model(self):
        """Main training loop using virtual epochs and DDP."""
        if self.is_master:
            print(f"\nStarting DDP training for {self.args.epochs} epochs on device {self.device}")
            self.visualizer = Visualizer('samples', self.device, self.writer)

        # Calculate steps per rank to reach a global samples_per_epoch
        world_size = dist.get_world_size()
        total_batches_per_epoch = self.args.samples_per_epoch // (self.args.batch_size * world_size)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if self.is_master:
            print(f"Mixed precision training is {'enabled' if self.args.use_amp else 'disabled'}. Using dtype {dtype} for mixed precision training.")

        for epoch in range(self.current_epoch, self.args.epochs):
            self.model.train()
            epoch_loss = 0.0
            self.train_sampler.set_epoch(epoch)

            if self.is_master:
                print(f"\n--- Epoch {epoch+1}/{self.args.epochs} ---")

            for batch_idx, (styled_input, target_hr) in enumerate(self.train_dataloader):
                # Break early to respect the virtual epoch limit
                if batch_idx >= total_batches_per_epoch:
                    break

                styled_input = styled_input.to(self.device, non_blocking=True)
                target_hr = target_hr.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True) # Zero gradients before forward pass

                with autocast(device_type=self.device.type, dtype=dtype, enabled=self.args.use_amp):
                    output_sr = self.model(styled_input)

                    # if not torch.isfinite(output_sr).all():
                    #     if self.is_master:
                    #         print(f"!!! NaN activations detected at batch {batch_idx}")
                    #     self.optimizer.zero_grad(set_to_none=True)
                    #     continue

                    loss = self.model.module.criterion(output_sr.float(), target_hr.float()) 

                    # if not torch.isfinite(loss):
                    #     if self.is_master:
                    #         print(f"!!! NaN loss at batch {batch_idx}. Model output finite: {torch.isfinite(output_sr).all()}")
                    #     # Skip this batch to prevent weight corruption
                    #     continue

                if self.args.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                current_step = (epoch * self.steps_per_epoch) + batch_idx

                if current_step < self.args.warmup_steps:
                    # Linear warmup: scale LR from 0 to max_lr
                    lr = (current_step / self.args.warmup_steps) * self.learning_rate
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    # After warmup, let the Cosine scheduler take over
                    self.scheduler.step(current_step)

                epoch_loss += loss.item()

                if self.is_master and (batch_idx + 1) % self.args.log_interval == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"    Batch {batch_idx+1}/{self.steps_per_epoch} | Loss: {loss.item():.6f} | LR: {current_lr:.8f}")
                    global_step = epoch * total_batches_per_epoch + batch_idx
                    self.writer.add_scalar('Loss/train_batch', loss.item(), global_step)
                    self.writer.add_scalar('LearningRate/batch', current_lr, global_step)

            # --- End of Epoch Training Aggregation ---
            avg_epoch_loss = epoch_loss / total_batches_per_epoch
            loss_tensor = torch.tensor([avg_epoch_loss], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            global_train_loss = loss_tensor.item() / world_size

            # --- Validation & Early Stopping ---
            if self.is_master:
                print(f"Epoch {epoch+1} training complete. Average Loss: {avg_epoch_loss:.6f}")
                print("Starting validation...")

            del styled_input, target_hr, output_sr, loss
            self.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(torch.cuda.memory_allocated() / 1024**3)
            print(torch.cuda.memory_reserved() / 1024**3)

            stop_training_tensor = torch.tensor([0], dtype=torch.int, device=self.device)
            global_val_loss = self.validate()

            if self.is_master:
                # Log metrics
                self.writer.add_scalar('Loss/train_epoch', global_train_loss, epoch)
                self.writer.add_scalar('Loss/val_epoch', global_val_loss, epoch)
                self.writer.add_scalar('LearningRate/epoch', self.optimizer.param_groups[0]['lr'], epoch)
                self.visualizer.log_epoch(self.model.module, epoch)
                print(f"Epoch {epoch+1} finished. Train Loss: {global_train_loss:.6f} | Val Loss: {global_val_loss:.6f}")

                # Check for improvement; determine if this is the best model (based on validation loss)
                should_stop, is_best = self.early_stopper.step(global_val_loss)
                if is_best:
                    self.early_stopper.counter = 0

                self.save_checkpoint(is_best, epoch, global_train_loss, global_val_loss)

                if should_stop:
                    print(f"Early stopping triggered after {self.args.early_stop_patience} epochs with no improvement.")
                    stop_training_tensor[0] = 1

            # Rank 1 will wait until Rank 0 finishes log_epoch.
            if dist.is_initialized():
                dist.barrier()

            # Broadcast stopping decision to all GPUs
            dist.broadcast(stop_training_tensor, src=0)

            if stop_training_tensor.item() == 1:
                break

        if self.is_master:
            self.writer.close()
            print("Training completed.")

# --------------------------------------------------------------------------------
# Argument Parsing
# --------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Super-Resolution Model Training.")

    # Data arguments
    parser.add_argument('--data_dir', nargs='+', required=True, 
                        help='Space-separated list of paths to directories containing generator output.')
    add_size_argument(parser, '--generator_crop_size', default="376 288", 
                      help='The \"W, H\" size of the images produced by generator.py (source crops). Default: 376 288')
    add_size_argument(parser, '--train_crop_size', default="376 288",
                      help='The \"W, H\" size of the random sub-crops used for actual training. Default: 376 288')
    parser.add_argument('--samples_per_epoch', type=int, default=50000, 
                        help='The number of samples the dataset reports for one epoch.')
    parser.add_argument('--shuffle_data', action='store_true', default=False,
                        help='Whether to shuffle data before splitting into train/val sets.')
    parser.add_argument('--print_model_layers', action='store_true', default=False,
                        help='If set, prints the model architecture layers.')

    # Model and training arguments
    parser.add_argument('--model_type', type=str, default='light', choices=['light', 'heavy'], help='Type of ResidualUNet model to use.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    add_size_argument(parser, '--batch_size', default=16, help='Batch size PER GPU for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Number of warmup steps for learning rate scheduler.')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='Beta1 for Adam optimizer.')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help='Beta2 for Adam optimizer.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of DataLoader workers per GPU.')
    parser.add_argument('--val_fraction', type=float, default=0.1)
    parser.add_argument('--val_limit', type=int, default=2000,
                        help='Maximum number of validation images to evaluate per epoch (rotating window).')
    parser.add_argument('--val_log_interval', type=int, default=50,
                        help='Log validation progress every N batches (per rank).')
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-delta", type=float, default=1e-4)
    parser.add_argument('--find_unused_parameters', action='store_true', default=False,
                        help='If True, set find_unused_parameters=True in DDP (needed if some model parameters are not used in every forward pass).')

    # Checkpointing and logging
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', 
                        help='Directory to save model checkpoints.')
    parser.add_argument('--load_checkpoint', type=str, default=None, 
                        help='Path to a checkpoint file to resume training from.')
    parser.add_argument('--reset_optimizer', action='store_true', 
                        help='If loading a checkpoint, only load weights and reset optimizer state/epoch count.')
    parser.add_argument('--log_dir', type=str, default='runs/sr_training', 
                        help='TensorBoard log directory.')
    parser.add_argument('--log_interval', type=int, default=50, 
                        help='How many batches to wait before logging training status.')

    # Performance arguments
    parser.add_argument('--use_amp', action='store_true', help='Use Automatic Mixed Precision (AMP).')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose model output.')
    parser.add_argument('--lores_only', action='store_true', help='Use lores only mode.')

    return parser.parse_args()

# --------------------------------------------------------------------------------
# Main Execution (DDP Launch Block)
# --------------------------------------------------------------------------------
def run_ddp_process(local_rank, args):
    """Initializes and runs a single DDP process."""
    # 1. Initialize Distributed Process
    # 'env://' uses the environment variables set by torchrun
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", device_id=local_rank)

    # Instantiate and Run Trainer
    try:
        trainer = Trainer(args, local_rank)
        trainer.train_model()
    except Exception as e:
        if local_rank == 0:
            print(f"A fatal error occurred on rank {local_rank}: {e}")
            import traceback
            traceback.print_exc()
    dist.destroy_process_group()

if __name__ == "__main__":
    args = parse_args()

    # DDP Launch Check (Checks for environment variables set by torchrun)
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK']) # Defined by torchrun
        run_ddp_process(local_rank, args)
    else:
        # Fallback for single process (for debugging or single-GPU use)
        # This requires manually setting the necessary environment variables
        print("Single GPU/CPU mode detected. Falling back to single-process DDP setup.")
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0' 
        run_ddp_process(0, args)
