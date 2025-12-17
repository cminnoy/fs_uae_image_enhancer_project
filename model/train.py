import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import sys
import time
import shutil
from typing import Tuple, Dict, Any

# --- DDP Imports ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# --------------------------------------------------------------------------------
# 1. Path Setup for Imports (Crucial for nested projects)
# This ensures Python can find 'model_residual_unet' (sibling) and 'srdataset' (root).
# --------------------------------------------------------------------------------
# Assuming train.py is inside the 'model' directory.
current_file_path = os.path.abspath(__file__)
# Go up one level (model) then up another level (fs_uae_image_enhancer_project)
project_root = os.path.dirname(os.path.dirname(current_file_path)) 
if project_root not in sys.path:
    sys.path.append(project_root)

# Import necessary components (corrected imports for script in 'model/' folder)
from model_residual_unet import get_model
from srdataset import SRDataset, gather_all_samples_from_directory, add_size_argument 


class Trainer:
    """
    Handles the DDP training and checkpointing logic for the ResidualUNet model.
    """
    # 4 space indentation
    def __init__(self, args, local_rank):
        self.args = args
        self.local_rank = local_rank # The assigned GPU index (0 or 1)
        
        # Set device based on local rank (e.g., cuda:0 or cuda:1)
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        
        # Only rank 0 handles logging and printing
        self.is_master = (self.local_rank == 0)
        if self.is_master:
            self.writer = SummaryWriter(log_dir=args.log_dir)
        
        self.scaler = GradScaler()
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.start_time = time.time()
        
        self.setup_data()
        self.setup_model_and_optimizer()

    def setup_data(self):
        # Data gathering and setup
        if self.is_master:
            print(f"Gathering samples from: {self.args.data_dir}")
        sample_pairs = gather_all_samples_from_directory(
            directory_path=self.args.data_dir,
            expected_crop_size=self.args.generator_crop_size,
            styles_to_include=None, # Include all styles for training
            verbose=1 if self.is_master else 0
        )
        
        # Determine training crop size from arguments (must be a tuple)
        if isinstance(self.args.train_crop_size, int):
            train_crop_size_tuple = (self.args.train_crop_size, self.args.train_crop_size)
        else:
            train_crop_size_tuple = self.args.train_crop_size
        
        self.train_dataset = SRDataset(
            sample_pairs_list=sample_pairs, 
            generator_output_crop_size=self.args.generator_crop_size, 
            train_crop_size=train_crop_size_tuple,
            num_samples=self.args.samples_per_epoch
        )
        
        # NEW: Use DistributedSampler
        self.train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=True
        )

        # DataLoader setup
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size, # This is batch size PER GPU
            shuffle=False, # Sampler handles shuffling
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=self.train_sampler # Pass the DistributedSampler
        )
        if self.is_master:
            world_size = dist.get_world_size()
            print(f"DataLoader ready with {len(self.train_dataset)} samples per epoch.")
            print(f"Effective Global Batch Size: {self.args.batch_size * world_size}")

    def setup_model_and_optimizer(self):
        # Model setup
        if self.is_master:
            print(f"Initializing model: {self.args.model_type}")
        self.model = get_model(self.args.model_type, verbose=self.args.verbose).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.args.learning_rate, 
            betas=(self.args.adam_beta1, self.args.adam_beta2)
        )

        # Load checkpoint if specified
        if self.args.load_checkpoint:
            self.load_checkpoint(self.args.load_checkpoint)
            
        # NEW: Wrap the model with DDP
        self.model = DDP(self.model, device_ids=[self.local_rank])

    def save_checkpoint(self, is_best: bool, epoch: int, loss: float):
        """Saves the model state, optimizer state, and training metrics."""
        if not self.is_master:
            return # Only rank 0 saves checkpoints

        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'best_loss': self.best_loss,
            'loss': loss,
            'model_state_dict': self.model.module.state_dict(), # <-- Use .module for unwrapped state_dict
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'args': self.args.__dict__,
            'time_elapsed': time.time() - self.start_time
        }

        filepath = os.path.join(self.args.checkpoint_dir, f'epoch_{epoch:04d}.pth')
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_filepath = os.path.join(self.args.checkpoint_dir, 'best_model.pth')
            shutil.copyfile(filepath, best_filepath)
            print(f"Saved best model checkpoint to {best_filepath}")
        
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
                self.current_epoch = checkpoint['epoch'] + 1 # Start next epoch
                self.best_loss = checkpoint.get('best_loss', float('inf'))
                self.start_time -= checkpoint.get('time_elapsed', 0.0)
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
        """Main training loop."""
        if self.is_master:
            print(f"\nStarting DDP training for {self.args.epochs} epochs on device {self.device}")
        
        for epoch in range(self.current_epoch, self.args.epochs):
            self.model.train()
            epoch_loss = 0.0
            
            # Set sampler epoch to ensure correct shuffling across processes
            self.train_sampler.set_epoch(epoch) 

            if self.is_master:
                print(f"\n--- Epoch {epoch+1}/{self.args.epochs} ---")
            
            for batch_idx, (styled_input, target_hr) in enumerate(self.train_dataloader):
                # Ensure data is on the correct device
                styled_input = styled_input.to(self.device, non_blocking=True)
                target_hr = target_hr.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad()
                
                # Mixed precision (AMP) context manager
                with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.args.use_amp):
                    output_sr = self.model(styled_input)
                    # Note: Model.module.criterion is used, as the DDP wrapper has no criterion attribute
                    loss = self.model.module.criterion(output_sr.float(), target_hr.float()) 
                
                # Backpropagation with GradScaler
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                current_loss = loss.item()
                epoch_loss += current_loss
                
                # Log batch metrics
                if self.is_master and (batch_idx + 1) % self.args.log_interval == 0:
                    avg_batch_loss = epoch_loss / (batch_idx + 1)
                    print(f"    Batch {batch_idx+1}/{len(self.train_dataloader)} | Loss: {current_loss:.6f} | Avg Loss: {avg_batch_loss:.6f}")
                    # Write to TensorBoard
                    global_step = epoch * len(self.train_dataloader) + batch_idx
                    self.writer.add_scalar('Loss/train_batch', current_loss, global_step)
            
            # Calculate average loss for the GPU
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            
            # NEW: Reduce (sum) the loss across all GPUs
            loss_tensor = torch.tensor([avg_epoch_loss]).to(self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            global_avg_loss = loss_tensor.item() / dist.get_world_size() 

            if self.is_master:
                # Log the reduced global loss
                self.writer.add_scalar('Loss/train_epoch', global_avg_loss, epoch)
                self.writer.add_scalar('LearningRate/epoch', self.optimizer.param_groups[0]['lr'], epoch)
                
                print(f"Epoch {epoch+1} finished. Global Average Loss: {global_avg_loss:.6f}")

                # Checkpoint management using the global average loss
                is_best = global_avg_loss < self.best_loss
                if is_best:
                    self.best_loss = global_avg_loss
                self.save_checkpoint(is_best, epoch, global_avg_loss)

            # Ensure all processes finish before proceeding to the next epoch
            dist.barrier()
            

        if self.is_master:
            self.writer.close()
            print("Training complete.")


# --------------------------------------------------------------------------------
# 2. Argument Parsing
# --------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Super-Resolution Model Training.")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Path to the directory containing generator output crops (e.g., /path/to/train).')
    add_size_argument(parser, '--generator_crop_size', default=(376, 288), 
                      help='The (W, H) size of the images produced by generator.py (source crops). Default: 376 288')
    add_size_argument(parser, '--train_crop_size', default=(376, 288), 
                      help='The (W, H) size of the random sub-crops used for actual training. Default: 128 128')
    parser.add_argument('--samples_per_epoch', type=int, default=50000, 
                        help='The number of samples the dataset reports for one epoch.')

    # Model and training arguments
    parser.add_argument('--model_type', type=str, default='light', 
                        choices=['light', 'heavy'], help='Type of ResidualUNet model to use.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    add_size_argument(parser, '--batch_size', default=16, help='Batch size PER GPU for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='Beta1 for Adam optimizer.')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help='Beta2 for Adam optimizer.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of DataLoader workers per GPU.')

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
    
    return parser.parse_args()


# --------------------------------------------------------------------------------
# 3. Main Execution (DDP Launch Block)
# --------------------------------------------------------------------------------
def run_ddp_process(local_rank, args):
    """Initializes and runs a single DDP process."""
    # 1. Initialize Distributed Process
    # 'env://' uses the environment variables set by torchrun
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", device_id=local_rank)

    # 2. Instantiate and Run Trainer
    try:
        trainer = Trainer(args, local_rank)
        trainer.train_model()
    except Exception as e:
        if local_rank == 0:
            print(f"A fatal error occurred on rank {local_rank}: {e}")
            
    # 3. Cleanup
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
