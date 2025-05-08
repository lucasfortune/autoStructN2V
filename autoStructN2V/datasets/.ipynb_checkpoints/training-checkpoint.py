# datasets/training.py
import random
import numpy as np
import torch

from .base import BaseNoiseDataset

class TrainingDataset(BaseNoiseDataset):
    """
    Dataset for training the first stage of Noise2Void with blind-spot masking.
    
    This dataset can be configured to either select patches randomly or use ROI-based
    selection. When using ROI selection, it can focus on either background regions
    or regions containing structures, making it flexible for both stages of training.
    
    Includes data augmentation with random flips and rotations.
    
    Args:
        image_paths (list): List of paths to input images.
        patch_size (int): Size of image patches to extract.
        kernel_size (int): Size of the kernel for blind-spot masking.
        mask_percentage (float): Percentage of pixels to mask in each patch (0.0 to 1.0).
        mask_strat (int): Masking strategy:
            0: Replace masked pixels with local mean
            1: Replace masked pixels with zeros
            2: Replace masked pixels with random values
        patches_per_image (int): Number of patches to extract per image.
        use_roi (bool, optional): Whether to use ROI-based patch selection. Defaults to True.
        scale_factor (float, optional): Factor to scale down images for ROI detection. 
            Only used if use_roi is True. Defaults to 0.25.
        roi_threshold (float, optional): Threshold for ROI detection. 
            Only used if use_roi is True. Defaults to 0.5.
        select_background (bool, optional): If True, selects background patches (above threshold).
            If False, selects structure patches (below threshold).
            Only used if use_roi is True. Defaults to True.
        use_augmentation (bool, optional): Whether to apply data augmentation. Defaults to True.
    """
    def __init__(self, image_paths, patch_size, kernel_size, mask, mask_percentage, mask_strat, 
                 prediction_kernel, patches_per_image=100, use_roi=True, scale_factor=0.25, 
                 roi_threshold=0.5, select_background=True, use_augmentation=True):
        super().__init__(image_paths, patch_size, patches_per_image)
        
        self.kernel_size = kernel_size
        self.mask_percentage = mask_percentage
        self.mask_strat = mask_strat
        self.prediction_kernel = prediction_kernel
        self.use_roi = use_roi
        self.scale_factor = scale_factor
        self.roi_threshold = roi_threshold
        self.select_background = select_background
        self.mask = mask
        self.use_augmentation = use_augmentation
        
        # Pre-compute ROI patches only if we're using ROI selection
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
    
    def apply_augmentation(self, image):
        """
        Apply random data augmentation to an image.
        
        Augmentations include:
        - Horizontal flip (50% chance)
        - Vertical flip (50% chance)
        - 90-degree rotations (25% chance for each of 90, 180, 270 degrees)
        
        Args:
            image (numpy.ndarray): Input image to augment
            
        Returns:
            numpy.ndarray: Augmented image
        """
        augmented = image.copy()
        
        # Random horizontal flip (50% chance)
        if random.random() > 0.5:
            augmented = np.fliplr(augmented)
            
        # Random vertical flip (50% chance)
        if random.random() > 0.5:
            augmented = np.flipud(augmented)
            
        # Random rotation (25% chance each for 90, 180, 270 degrees)
        k = random.choice([0, 1, 2, 3])  # 0: no rotation, 1: 90°, 2: 180°, 3: 270°
        if k > 0:
            augmented = np.rot90(augmented, k=k)
            
        return augmented
        
    def __getitem__(self, idx):
        """
        Get a training sample consisting of an input patch, target patch, and mask.
        The input patch has some pixels masked according to the noise2void strategy.
        Applies random augmentation if enabled.
        """
        # Calculate which image and which patch within that image
        image_idx = idx // self.patches_per_image
        
        # Load and normalize the full image
        img_array = self.load_image(image_idx)
        
        # Apply augmentation to the full image if enabled
        if self.use_augmentation:
            img_array = self.apply_augmentation(img_array)
        
        # Select patch location based on configuration
        if self.use_roi and self.roi_patches[image_idx]:
            # Choose random coordinates from pre-computed ROI patches
            top, left = random.choice(self.roi_patches[image_idx])
            
            # Check if the selected patch would go out of bounds after augmentation
            # (e.g., if rotation changed the image dimensions)
            h, w = img_array.shape
            if top + self.patch_size > h or left + self.patch_size > w:
                # Fall back to random selection if out of bounds
                top = np.random.randint(0, h - self.patch_size)
                left = np.random.randint(0, w - self.patch_size)
        else:
            # Random selection if not using ROI or if no ROI patches found
            h, w = img_array.shape
            top = np.random.randint(0, h - self.patch_size)
            left = np.random.randint(0, w - self.patch_size)
            
        # Extract the patch from the full image
        patch = img_array[top:top+self.patch_size, left:left+self.patch_size]
        
        # Apply masking directly to create input
        input_patch, mask = self.apply_mask(patch, 
                                  mask_percentage=self.mask_percentage,
                                  mask_strat=self.mask_strat,
                                  kernel_size=self.kernel_size,
                                  mask=self.mask)
        
        # Create target patch (original, unmasked patch)
        target_patch = patch.copy()
        
        # Convert numpy arrays to PyTorch tensors
        input_tensor = self.to_tensor(input_patch)
        target_tensor = self.to_tensor(target_patch)
        mask_tensor = torch.from_numpy(self.prediction_kernel).float() 
        
        return input_tensor, target_tensor, mask_tensor
        
    def apply_mask(self, patch, mask_percentage, mask_strat, kernel_size, mask):
        """
        Apply masking kernel to an image patch using various strategies.
        
        Args:
            patch (numpy.ndarray): Input image patch
            mask_percentage (float): Percentage of patch to mask (0-100)
            mask_strat (int): Masking strategy:
                0: Replace with local mean of unmasked neighbors
                1: Replace with zero
                2: Replace with random unmasked value
            kernel_size (int): Size of mask kernel for non-structured masking
            mask (numpy.ndarray): Boolean array showing masked positions
            
        Returns:
            tuple:
                masked_patch (numpy.ndarray): Patch with masking applied
                mask (numpy.ndarray): Boolean array showing masked positions
        """
        h, w = patch.shape
        
        if mask.shape != patch.shape:
            raise ValueError("Mask shape doesn't match patch shape")
            
        masked_patch = patch.copy()
        
        # Vectorized implementation for zero and random strategies
        if mask_strat == 1:  # Zero: pixels are black
            masked_patch[mask] = 0
        elif mask_strat == 2:  # Random
            unmasked_values = patch[~mask]
            masked_patch[mask] = np.random.choice(unmasked_values, size=np.sum(mask))
        else:  # Local mean (needs loop due to neighborhood operations)
            for i in range(h):
                for j in range(w):
                    if mask[i, j]:
                        # Get neighborhood indices
                        i_start, i_end = max(0, i-1), min(h, i+2)
                        j_start, j_end = max(0, j-1), min(w, j+2)
                        
                        neighborhood = patch[i_start:i_end, j_start:j_end]
                        neighborhood_mask = mask[i_start:i_end, j_start:j_end]
                        unmasked_values = neighborhood[~neighborhood_mask]
                        
                        if len(unmasked_values) > 0:
                            masked_patch[i, j] = np.mean(unmasked_values)
                        else:
                            # Fallback to random unmasked value if no valid neighbors
                            unmasked_values = patch[~mask]
                            masked_patch[i, j] = np.random.choice(unmasked_values)
        
        return masked_patch, mask