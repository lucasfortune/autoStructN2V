# trainers/auto_struct_n2v.py
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from .base import BaseTrainer
from .callbacks import EarlyStopping
from ..utils.patching import image_to_patches, patches_to_image

class AutoStructN2VTrainer(BaseTrainer):
    """
    Trainer for the AutoStructN2V model.
    
    This trainer implements the 2-stage autoStructNoise2Void approach, with stage 1
    focusing on standard Noise2Void denoising and stage 2 on structured denoising.
    
    Args:
        model (nn.Module): Model to train
        optimizer (torch.optim.Optimizer): Optimizer for training
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
        device (torch.device): Device to use for training
        hparams (dict): Hyperparameters for training
        stage (str): Training stage ('stage1' or 'stage2')
        experiment_name (str): Name of the experiment for logging
    """
    def __init__(self, model, optimizer, scheduler, device, hparams, stage, experiment_name='experiment'):
        super().__init__(model, optimizer, scheduler, device, hparams)
        
        self.stage = stage
        if stage not in ['stage1', 'stage2']:
            raise ValueError(f"Invalid stage: {stage}. Must be 'stage1' or 'stage2'.")
            
        self.experiment_name = experiment_name
        
        # Set up logging based on stage
        stage_name = 'n2v' if stage == 'stage1' else 'structn2v'
        self.log_dir = os.path.join(experiment_name, stage_name, self.current_time)
        self.writer = SummaryWriter(self.log_dir)
        
        # Log hyperparameters
        self.writer.add_text("Hyperparameters", 
                             "\n".join(f"{k}: {v}" for k, v in hparams.items()))
    
    def train(self, train_loader, val_loader, test_loader=None):
        """
        Main training loop for AutoStructN2V.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            test_loader (DataLoader, optional): Test data loader. Required for stage2. Defaults to None.
            
        Returns:
            For stage1: numpy.ndarray - Custom mask for stage2
            For stage2: None
        """
        # Validate inputs based on stage
        if self.stage == 'stage2' and test_loader is None:
            raise ValueError("test_loader is required for stage2 training")

        # Set up early stopping
        early_stopping = EarlyStopping(
            patience=self.hparams.get('early_stopping_patience', 10),
            min_delta=self.hparams.get('early_stopping_min_delta', 0.001)
        )
        
        # Path to save best model
        best_model_path = os.path.join(self.log_dir, 'best_model.pth')
        
        # Main training loop
        print(f"Training {self.stage} model...")
        with tqdm(total=self.hparams['num_epochs'], desc="Training Progress", ncols=100) as pbar:
            for epoch in range(self.hparams['num_epochs']):
                # Training phase
                train_loss = self.train_epoch(train_loader)
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                
                # Validation phase
                val_loss = self.validate_epoch(val_loader)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

                # Log test images periodically
                if test_loader and epoch % 5 == 0:
                    patch_size = self.hparams['n2v_patch_size'] if self.stage == 'stage1' else self.hparams['structn2v_patch_size']
                    stride = patch_size // 2
                    self.log_test_images(test_loader, self.writer, epoch, patch_size, stride)
                
                # Model saving and early stopping
                if val_loss < early_stopping.best_loss:
                    self.save_checkpoint(best_model_path)
                    
                if early_stopping(val_loss) and self.hparams.get('early_stopping', True):
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
                    
                pbar.update(1)
        
        print(f"Training completed. Final losses - Training: {train_loss:.4f}, Validation: {val_loss:.4f}")
        
        # Load best model for inference
        self.load_checkpoint(best_model_path)
        
        # Stage-specific post-training actions
        if self.stage == 'stage1':
            print("Using Model to denoise patches for Stage 2...")
            denoised_patches = self.create_denoised_patches(train_loader)
            self.writer.close()
            return denoised_patches
        else:  # stage2
            print("Starting testing phase...")
            patch_size = self.hparams['structn2v_patch_size']
            stride = patch_size // 2
            test_loss = self.test_model(test_loader, self.writer, patch_size, stride)
            print(f"Testing completed. Test Loss: {test_loss:.4f}")
            self.writer.close()
            return None
    
    def calculate_loss(self, pred, target, mask):
        """
        Calculate the loss for AutoStructN2V training, supporting both stages.
        
        Args:
            pred (torch.Tensor): Predicted tensor
            target (torch.Tensor): Target tensor
            mask (torch.Tensor): Mask tensor
            
        Returns:
            torch.Tensor: Calculated loss
        """
        # Validate input shapes
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
        
        # Calculate pixel-wise losses
        pixel_losses = nn.MSELoss(reduction='none')(pred, target)
        
        # Apply mask to only consider valid pixels
        loss = (pixel_losses * mask).mean()

        return loss
    
    def create_denoised_patches(self, data_loader):
        """
        Create denoised patches using the trained stage1 model.
        
        These patches will be used to create masks for the stage2 model.
        
        Args:
            data_loader (DataLoader): Data loader with patches
            
        Returns:
            numpy.ndarray: Denoised patches
        """
        self.model.eval()
        denoised_patches = []
        
        with torch.no_grad():
            for inputs, targets, _ in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                
                # Move outputs to CPU and convert to numpy
                denoised_np = outputs.cpu().numpy()
                denoised_patches.extend([patch for patch in denoised_np])
                
                # Limit the number of patches to process
                if len(denoised_patches) >= 1000:
                    break
        
        return np.array(denoised_patches)