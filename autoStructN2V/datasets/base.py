# datasets/base.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from ..utils.image import load_and_normalize_image, load_and_preprocess_image
from ..utils.patching import find_roi_patches

class BaseNoiseDataset(Dataset):
    """
    Base dataset class that implements common functionality for all Noise2Void datasets.
    
    This class serves as a foundation for different types of Noise2Void datasets (training,
    validation, and testing). It handles basic operations like image loading, tensor
    conversion, and dataset length calculation. The class is designed to be flexible,
    supporting both patch-based processing (for training/validation) and full-image
    processing (for testing).

    Args:
        image_paths (list): List of paths to input images.
        patch_size (int, optional): Size of image patches to extract. Required for 
            training and validation datasets, but not for test datasets.
        patches_per_image (int, optional): Number of patches to extract per image.
            Required for training and validation datasets, but not for test datasets.

    Methods:
        __len__: Returns dataset length (either number of total patches or number of images)
        load_image: Helper method to load and normalize an image
        __getitem__: Must be implemented by child classes
    """
    def __init__(self, image_paths, patch_size=None, patches_per_image=None):
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        """
        Calculate the length of the dataset.
        
        Returns:
            int: For training/validation datasets, returns total number of patches
                 (images × patches_per_image). For test datasets, returns number of images.
        """
        if self.patches_per_image is None:
            return len(self.image_paths)
        return len(self.image_paths) * self.patches_per_image

    def load_image(self, idx):
        """
        Load and normalize an image from the dataset.
        
        Args:
            idx (int): Index of the image to load.
            
        Returns:
            numpy.ndarray: Normalized image array.
        """
        img_path = self.image_paths[idx]
        return load_and_normalize_image(img_path)

    def preprocess_for_roi(self, img_path, scale_factor):
        """
        Preprocess an image for ROI detection.
        
        Args:
            img_path (str): Path to the image file
            scale_factor (float): Factor to scale down the image
            
        Returns:
            numpy.ndarray: Preprocessed image ready for ROI detection
        """
        return load_and_preprocess_image(img_path, scale_factor=scale_factor)
    
    def get_roi_patches(self, preprocessed_img, patch_size, threshold, above_threshold, scale_factor):
        """
        Get ROI patch coordinates using the improved find_roi_patches function.
        
        Args:
            preprocessed_img (numpy.ndarray): Preprocessed image array
            patch_size (int): Size of patches to extract
            threshold (float): Threshold for ROI detection
            above_threshold (bool): Whether to select patches above threshold
            scale_factor (float): Scale factor for coordinate conversion
            
        Returns:
            list: List of patch coordinates (top, left)
        """
        return find_roi_patches(
            preprocessed_img, 
            patch_size, 
            threshold=threshold,
            above_threshold=above_threshold,
            scale_factor=scale_factor
        )