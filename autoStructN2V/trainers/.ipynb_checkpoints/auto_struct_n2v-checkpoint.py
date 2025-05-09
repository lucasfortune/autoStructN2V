# autoStructN2V/trainers/auto_struct_n2v.py
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import torchvision.utils as vutils  # Add this import

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
        os.makedirs(experiment_name, exist_ok=True)  # Ensure experiment directory exists
        self.log_dir = os.path.join(experiment_name, stage_name, self.current_time)
        self.writer = SummaryWriter(self.log_dir)
        
        # Log hyperparameters - convert dict to string for TensorBoard
        hparams_str = self._format_hparams_for_logging(hparams)
        self.writer.add_text("Hyperparameters", hparams_str)
    
    def _format_hparams_for_logging(self, hparams):
        """Format hyperparameters for TensorBoard logging."""
        lines = []
        for k, v in hparams.items():
            if isinstance(v, dict):
                lines.append(f"{k}:")
                for sub_k, sub_v in v.items():
                    lines.append(f"  {sub_k}: {sub_v}")
            else:
                lines.append(f"{k}: {v}")
        return "\n".join(lines)
    
    def _get_param(self, param_name, default_value=None):
        """
        Get a parameter from hparams, supporting both structured and flat config formats.
        
        Args:
            param_name (str): Parameter name without prefix
            default_value: Default value if parameter is not found
            
        Returns:
            Parameter value
        """
        # Check structured format first
        if self.stage == 'stage1':
            if 'stage1' in self.hparams and param_name in self.hparams['stage1']:
                return self.hparams['stage1'][param_name]
            # Fall back to flat format with n2v_ prefix
            return self.hparams.get(f'n2v_{param_name}', default_value)
        else:  # stage2
            if 'stage2' in self.hparams and param_name in self.hparams['stage2']:
                return self.hparams['stage2'][param_name]
            # Fall back to flat format with structn2v_ prefix
            return self.hparams.get(f'structn2v_{param_name}', default_value)
    
    def train(self, train_loader, val_loader, test_loader=None):
        """
        Main training loop for AutoStructN2V.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            test_loader (DataLoader, optional): Test data loader. Required for stage2. Defaults to None.
            
        Returns:
            For stage1: numpy.ndarray - denoised patches for custom mast creation for stage2
            For stage2: None
        """
        # Validate inputs based on stage
        if self.stage == 'stage2' and test_loader is None:
            raise ValueError("test_loader is required for stage2 training")

        # Set up early stopping
        patience = self.hparams.get('early_stopping_patience', 10)
        min_delta = self.hparams.get('early_stopping_min_delta', 0.001)
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
        
        # Path to save best model
        os.makedirs(self.log_dir, exist_ok=True)  # Ensure log directory exists
        best_model_path = os.path.join(self.log_dir, 'best_model.pth')
        
        # Get number of epochs - check in both locations
        num_epochs = self.hparams.get('num_epochs', 100)
        
        # Main training loop
        print(f"Training {self.stage} model...")
        with tqdm(total=num_epochs, desc="Training Progress", ncols=100) as pbar:
            for epoch in range(num_epochs):
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
                    # Get patch_size from config using helper method
                    patch_size = self._get_param('patch_size', 64)
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
            patch_size = self._get_param('patch_size', 64)
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
            # Get max patches to process from config
            max_patches = self._get_param('max_denoised_patches', 1000)
            
            for inputs, targets, _ in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                
                # Move outputs to CPU and convert to numpy
                denoised_np = outputs.cpu().numpy()
                denoised_patches.extend([patch for patch in denoised_np])
                
                # Limit the number of patches to process
                if len(denoised_patches) >= max_patches:
                    break
        
        return np.array(denoised_patches)
    
    def log_test_images(self, test_loader, writer, epoch, patch_size, stride):
        """
        Log test images during training.
        
        Args:
            test_loader (DataLoader): Test data loader
            writer (SummaryWriter): TensorBoard writer
            epoch (int): Current epoch
            patch_size (int): Size of patches for processing
            stride (int): Stride for patch extraction
        """
        self.model.eval()
        all_images = []
        
        with torch.no_grad():
            for inputs, targets, masks in test_loader:
                inputs = inputs.to(self.device)
                
                # Process image patches
                input_patches = image_to_patches(inputs[0], patch_size, stride)
                output_patches = []
                
                for patch in input_patches:
                    output = self.model(patch.unsqueeze(0))
                    output_patches.append(output.squeeze(0))
                
                output_patches = torch.stack(output_patches)
                outputs = patches_to_image(output_patches, inputs[0].shape, patch_size, stride)
                
                # Store results
                inputs = torch.clamp(inputs.cpu(), 0, 1)
                outputs = torch.clamp(outputs.cpu().unsqueeze(0), 0, 1)
                combined = torch.cat((inputs, outputs), dim=3)
                all_images.append(combined)
        
        if all_images:
            all_images = torch.cat(all_images, dim=0)
            
            # Log test results
            writer.add_image(f'Test/Epoch_{epoch}',
                            vutils.make_grid(all_images, nrow=1, padding=2, normalize=False),
                            0)
    
    def test_model(self, test_loader, writer, patch_size, stride):
        """
        Test model on full images.
        
        Args:
            test_loader (DataLoader): Test data loader
            writer (SummaryWriter): TensorBoard writer
            patch_size (int): Size of patches for processing
            stride (int): Stride for patch extraction
            
        Returns:
            float: Average test loss
        """
        self.model.eval()
        test_loss = 0.0
        all_images = []
        
        with torch.no_grad():
            for inputs, targets, masks in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                masks = masks.to(self.device)
                
                # Process image patches
                input_patches = image_to_patches(inputs[0], patch_size, stride)
                output_patches = []
                
                for patch in input_patches:
                    output = self.model(patch.unsqueeze(0))
                    output_patches.append(output.squeeze(0))
                
                output_patches = torch.stack(output_patches)
                outputs = patches_to_image(output_patches, inputs[0].shape, patch_size, stride)
                
                # Calculate and log loss
                loss = self.calculate_loss(outputs.unsqueeze(0), targets, masks)
                test_loss += loss.item()
                
                # Store results
                inputs = torch.clamp(inputs.cpu(), 0, 1)
                outputs = torch.clamp(outputs.cpu().unsqueeze(0), 0, 1)
                combined = torch.cat((inputs, outputs), dim=3)
                all_images.append(combined)
        
        if all_images:
            test_loss /= len(test_loader)
            all_images = torch.cat(all_images, dim=0)
            
            # Log test results
            writer.add_scalar('Loss/test', test_loss, 0)
            writer.add_image('Test/Input_and_Output_Images',
                            vutils.make_grid(all_images, nrow=1, padding=2, normalize=False),
                            0)
        else:
            test_loss = 0.0
        
        return test_loss