# datasets/testing.py
import torch

from .base import BaseNoiseDataset

class TestDataset(BaseNoiseDataset):
    """
    Dataset for testing trained Noise2Void models on complete images.
    
    This dataset handles full images rather than patches, allowing the model
    to be evaluated on complete images. It's designed for final evaluation
    and actual denoising tasks on new images.

    Args:
        image_paths (list): List of paths to test images.

    Unlike training and validation datasets, this dataset:
    1. Processes entire images instead of patches
    2. Returns one image per __getitem__ call
    3. Uses the same tensor for input and target (for evaluation)
    4. Provides a full mask (all ones) since no masking is needed for testing
    """
    def __getitem__(self, idx):
        """
        Get a test sample consisting of a full image.
        
        Returns:
            tuple: (input_tensor, target_tensor, mask_tensor) where all tensors
                are identical (no masking for testing)
        """
        img_array = self.load_image(idx)
        input_tensor = self.to_tensor(img_array)
        return input_tensor, input_tensor.clone(), torch.ones_like(input_tensor)