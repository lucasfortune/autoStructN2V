# datasets/validation.py
import random
import numpy as np
import torch

from .base import BaseNoiseDataset

class ValidationDataset(BaseNoiseDataset):
    """
    Dataset for validating Noise2Void models during training.
    
    This dataset can be configured to select patches either randomly or using
    ROI detection, matching the behavior of the training dataset. For the first
    stage (N2V), it can focus on background regions, while for the second stage
    (StructN2V), it can focus on regions containing structures.
    
    Args:
        image_paths (list): List of paths to validation images.
        patch_size (int): Size of image patches to extract.
        patches_per_image (int): Number of patches to extract per image.
        use_roi (bool, optional): Whether to use ROI selection. Defaults to False.
        scale_factor (float, optional): Factor to scale down images for ROI detection.
            Only used if use_roi is True. Defaults to 0.25.
        roi_threshold (float, optional): Threshold for ROI detection.
            Only used if use_roi is True. Defaults to 0.5.
        select_background (bool, optional): If True, selects background patches
            (above threshold). If False, selects structure patches (below threshold).
            Only used if use_roi is True. Defaults to True.
    """
    def __init__(self, image_paths, patch_size, patches_per_image, 
                 use_roi=False, scale_factor=0.25, roi_threshold=0.5, 
                 select_background=True):
        super().__init__(image_paths, patch_size, patches_per_image)
        self.use_roi = use_roi
        self.scale_factor = scale_factor
        self.roi_threshold = roi_threshold
        self.select_background = select_background
        
        # Pre-compute ROI patches if using ROI selection
        self.roi_patches = []
        if self.use_roi:
            for img_path in image_paths:
                preprocessed_img = self.preprocess_for_roi(img_path, scale_factor)
                patches = self.get_roi_patches(
                    preprocessed_img,
                    patch_size,
                    threshold=roi_threshold,
                    above_threshold=select_background,  # True for background, False for structures
                    scale_factor=scale_factor
                )
                self.roi_patches.append(patches)

    def __getitem__(self, idx):
        """
        Get a validation sample.
        
        Returns both the input patch and target patch as identical tensors,
        since no masking is applied for validation.
        """
        image_idx = idx // self.patches_per_image
        img_array = self.load_image(image_idx)
        
        if self.use_roi and self.roi_patches[image_idx]:
            # Choose random coordinates from pre-computed ROI patches
            top, left = random.choice(self.roi_patches[image_idx])
        else:
            # Random selection if not using ROI or if no ROI patches found
            h, w = img_array.shape
            top = np.random.randint(0, h - self.patch_size)
            left = np.random.randint(0, w - self.patch_size)
            
        # Extract the patch
        patch = img_array[top:top+self.patch_size, left:left+self.patch_size]
        
        # Convert to tensor and return
        input_tensor = self.to_tensor(patch)
        return input_tensor, input_tensor.clone(), torch.ones_like(input_tensor)