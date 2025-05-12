# trainers/base.py
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import torchvision.utils as vutils

from ..utils.patching import image_to_patches, patches_to_image
from .callbacks import EarlyStopping

class BaseTrainer:
    """
    Base trainer class providing common training functionality.
    
    This class provides the foundation for specialized trainers,
    handling common operations like training loops, validation,
    and metric logging.
    
    Args:
        model (nn.Module): Model to train
        optimizer (torch.optim.Optimizer): Optimizer for training
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
        device (torch.device): Device to use for training
        hparams (dict): Hyperparameters for training
        log_dir (str, optional): Directory for saving logs. Defaults to 'logs'.
    """
    def __init__(self, model, optimizer, scheduler, device, hparams, log_dir='logs'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.hparams = hparams
        
        # Create timestamp for logging
        self.current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = log_dir
        
        # Move model to device
        self.model.to(self.device)
    
    def train(self, train_loader, val_loader, test_loader=None):
        """
        Main training loop with validation and testing.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            test_loader (DataLoader, optional): Test data loader. Defaults to None.
            
        Returns:
            nn.Module: Trained model
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def train_epoch(self, train_loader):
        """
        Run one epoch of training.
        
        Args:
            train_loader (DataLoader): Training data loader
            
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        
        for inputs, targets, masks in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            masks = masks.to(self.device).unsqueeze(1)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.calculate_loss(outputs, targets, masks)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate_epoch(self, val_loader):
        """
        Run one epoch of validation.
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            float: Average validation loss for the epoch
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets, masks in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.calculate_loss(outputs, targets, masks)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def calculate_loss(self, pred, target, mask):
        """
        Calculate the loss for model training.
        
        Args:
            pred (torch.Tensor): Predicted tensor of shape (batch_size, channels, height, width)
            target (torch.Tensor): Target tensor of shape (batch_size, channels, height, width)
            mask (torch.Tensor): Mask tensor indicating which pixels to include in loss
            
        Returns:
            torch.Tensor: Calculated loss value
        """
        # Validate input shapes
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
            
        # Calculate MSE without reduction to keep per-pixel losses
        pixel_losses = nn.MSELoss(reduction='none')(pred, target)
        
        # Apply mask to only consider valid pixels
        loss = (pixel_losses * mask).mean()

        return loss
    
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
                original_shape = inputs[0].shape
                
                # Calculate required padding
                h, w = original_shape[1:]
                pad_h = (patch_size - h % stride) % stride
                pad_w = (patch_size - w % stride) % stride
                
                # Apply padding if needed
                if pad_h > 0 or pad_w > 0:
                    padded_inputs = torch.nn.functional.pad(inputs, (0, pad_w, 0, pad_h), mode='reflect')
                else:
                    padded_inputs = inputs
                
                # Process image patches
                input_patches = image_to_patches(padded_inputs[0], patch_size, stride)
                output_patches = []
                
                for patch in input_patches:
                    output = self.model(patch.unsqueeze(0))
                    output_patches.append(output.squeeze(0))
                
                output_patches = torch.stack(output_patches)
                outputs = patches_to_image(output_patches, padded_inputs[0].shape, patch_size, stride)
                
                # Crop back to original size
                outputs = outputs[:, :original_shape[1], :original_shape[2]]
                
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
                original_shape = inputs[0].shape
                
                # Calculate required padding
                h, w = original_shape[1:]
                pad_h = (patch_size - h % stride) % stride
                pad_w = (patch_size - w % stride) % stride
                
                # Apply padding if needed
                if pad_h > 0 or pad_w > 0:
                    padded_inputs = torch.nn.functional.pad(inputs, (0, pad_w, 0, pad_h), mode='reflect')
                else:
                    padded_inputs = inputs
                
                # Process image patches
                input_patches = image_to_patches(padded_inputs[0], patch_size, stride)
                output_patches = []
                
                for patch in input_patches:
                    output = self.model(patch.unsqueeze(0))
                    output_patches.append(output.squeeze(0))
                
                output_patches = torch.stack(output_patches)
                outputs = patches_to_image(output_patches, padded_inputs[0].shape, patch_size, stride)
                
                # Crop back to original size
                outputs = outputs[:, :original_shape[1], :original_shape[2]]
                
                # Calculate and log loss
                outputs_for_loss = outputs.unsqueeze(0)
                loss = self.calculate_loss(outputs_for_loss, targets, masks)
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
    
    def save_checkpoint(self, path):
        """
        Save a model checkpoint.
        
        Args:
            path (str): Path to save the checkpoint
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'hparams': self.hparams
        }, path)
        
    def load_checkpoint(self, path):
        """
        Load a model checkpoint.
        
        Args:
            path (str): Path to the checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])