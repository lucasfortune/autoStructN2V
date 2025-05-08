# models/auto_struct_n2v.py
import torch
import torch.nn as nn

from .unet import FlexibleUNet

class AutoStructN2VModel(nn.Module):
    """
    AutoStructN2V model implementing the 2-stage autoStructNoise2Void approach.
    
    This model implements a specialized denoising pipeline that combines single-pixel
    denoising (Stage 1: N2V) with structured denoising (Stage 2: StructN2V) into a
    unified approach. Both stages utilize the same UNet architecture but differ in
    how they are trained.
    
    Args:
        features (int): Number of features in the first UNet layer
        num_layers (int): Number of down/up-sampling layers in the UNet
        in_channels (int, optional): Number of input channels. Defaults to 1.
        out_channels (int, optional): Number of output channels. Defaults to 1.
        stage (str, optional): The stage this model represents ('stage1' or 'stage2'). 
            Defaults to None.
    """
    def __init__(self, features, num_layers, in_channels=1, out_channels=1, stage=None):
        super(AutoStructN2VModel, self).__init__()
        
        # The core UNet architecture
        self.unet = FlexibleUNet(
            features=features,
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels
        )
        
        # Track which stage this model represents
        self.stage = stage
    
    def forward(self, x):
        """
        Forward pass through the AutoStructN2V model.
        
        Args:
            x (torch.Tensor): Input noisy tensor of shape (batch_size, in_channels, H, W)
            
        Returns:
            torch.Tensor: Denoised output tensor of shape (batch_size, out_channels, H, W)
        """
        # Both stages use the same UNet architecture for forward pass
        return self.unet(x)
    
    @classmethod
    def create_stage1_model(cls, features, num_layers, in_channels=1, out_channels=1):
        """
        Factory method to create a Stage 1 model (standard N2V).
        
        Args:
            features (int): Number of features in the first UNet layer
            num_layers (int): Number of down/up-sampling layers in the UNet
            in_channels (int, optional): Number of input channels. Defaults to 1.
            out_channels (int, optional): Number of output channels. Defaults to 1.
            
        Returns:
            AutoStructN2VModel: Model configured for Stage 1 denoising
        """
        return cls(features, num_layers, in_channels, out_channels, stage='stage1')
    
    @classmethod
    def create_stage2_model(cls, features, num_layers, in_channels=1, out_channels=1):
        """
        Factory method to create a Stage 2 model (structured N2V).
        
        Args:
            features (int): Number of features in the first UNet layer
            num_layers (int): Number of down/up-sampling layers in the UNet
            in_channels (int, optional): Number of input channels. Defaults to 1.
            out_channels (int, optional): Number of output channels. Defaults to 1.
            
        Returns:
            AutoStructN2VModel: Model configured for Stage 2 denoising
        """
        return cls(features, num_layers, in_channels, out_channels, stage='stage2')