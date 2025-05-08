# utils/patching.py
import torch
import numpy as np

def image_to_patches(image, patch_size, stride):
    """
    Convert an image into a series of overlapping patches for processing.
    
    This function slides a window across the image, extracting square patches.
    The amount of overlap between patches is controlled by the stride parameter.
    
    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W)
        patch_size (int): Size of each square patch
        stride (int): Distance between patch starting positions.
            stride < patch_size creates overlapping patches
            
    Returns:
        torch.Tensor: Tensor of patches of shape (N, C, patch_size, patch_size)
            where N is the number of patches
            
    Raises:
        ValueError: If patch_size or stride are invalid, or if image is too small
    """
    if len(image.shape) != 3:
        raise ValueError("Expected image tensor of shape (C, H, W)")
    if patch_size > min(image.shape[1:]):
        raise ValueError("patch_size cannot be larger than image dimensions")
    if stride <= 0:
        raise ValueError("stride must be positive")
        
    patches = []
    h, w = image.shape[1:]
    
    # Calculate valid ranges for patch extraction
    h_range = range(0, h - patch_size + 1, stride)
    w_range = range(0, w - patch_size + 1, stride)
    
    # Extract patches using tensor operations
    for i in h_range:
        for j in w_range:
            patch = image[:, i:i+patch_size, j:j+patch_size]
            patches.append(patch)
            
    return torch.stack(patches)

def patches_to_image(patches, image_size, patch_size, stride):
    """
    Reconstruct an image from overlapping patches using weighted averaging.
    
    This function places each patch back in its original position and combines
    overlapping regions using a weighted average. The weights give more importance
    to the center of each patch to reduce edge artifacts.
    
    Args:
        patches (torch.Tensor): Tensor of patches of shape (N, C, patch_size, patch_size)
        image_size (tuple): Original image size as (C, H, W)
        patch_size (int): Size of each square patch
        stride (int): Stride used when extracting patches
        
    Returns:
        torch.Tensor: Reconstructed image of shape (C, H, W)
        
    Raises:
        ValueError: If input dimensions don't match or parameters are invalid
    """
    device = patches.device
    channels = patches.shape[1]
    
    # Validate inputs
    if len(image_size) != 3:
        raise ValueError("image_size should be (C, H, W)")
    if channels != image_size[0]:
        raise ValueError("Number of channels in patches and image_size don't match")
    if patch_size > min(image_size[1:]):
        raise ValueError("patch_size cannot be larger than image dimensions")
    
    # Initialize output tensors
    output = torch.zeros((channels, *image_size[1:]), device=device)
    weight_sum = torch.zeros((channels, *image_size[1:]), device=device)
    
    # Create and expand weight mask for all channels
    weight_mask = create_weight_mask(patch_size).to(device)
    weight_mask = weight_mask.unsqueeze(0).repeat(channels, 1, 1)
    
    # Reconstruct image
    patch_index = 0
    h_range = range(0, image_size[1] - patch_size + 1, stride)
    w_range = range(0, image_size[2] - patch_size + 1, stride)
    
    for i in h_range:
        for j in w_range:
            output[:, i:i+patch_size, j:j+patch_size] += patches[patch_index] * weight_mask
            weight_sum[:, i:i+patch_size, j:j+patch_size] += weight_mask
            patch_index += 1
    
    # Normalize by weights, avoiding division by zero
    eps = torch.tensor(1e-8, device=device)
    weight_sum = torch.maximum(weight_sum, eps)
    reconstructed = output / weight_sum
    
    return reconstructed

def create_weight_mask(patch_size, alpha=1):
    """
    Create a weight mask for blending overlapping patches to avoid edge artifacts.
    
    This function creates a 2D weight mask where pixels near the center of the patch
    have higher weights than those near the edges. When patches are combined, this
    weighting reduces visible seams between patches.
    
    Args:
        patch_size (int): Size of the square patch
        alpha (float, optional): Controls how quickly weights fall off from center.
            Higher values create sharper falloff. Defaults to 1.
            
    Returns:
        torch.Tensor: 2D weight mask of shape (patch_size, patch_size)
        
    Raises:
        ValueError: If patch_size is not positive or alpha is negative
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
        
    center = patch_size // 2
    x = torch.arange(patch_size, dtype=torch.float32)
    
    # Create linear falloff from center
    weight_1d = 1 - torch.abs(x - center) / center
    
    # Apply power function for non-linear falloff
    weight_1d = torch.pow(weight_1d.clamp(min=0), alpha)
    
    # Create 2D mask through outer product
    weight_2d = weight_1d.unsqueeze(0) * weight_1d.unsqueeze(1)
    
    return weight_2d

def find_roi_patches(img, patch_size, threshold=0.2, above_threshold=False, 
                     max_patches=10000, scale_factor=1.0, overlap=0.5):
    """
    Find regions of interest (ROI) in the image for patch extraction.
    
    This function scans the image with a sliding window to find patches that meet
    specified intensity criteria. It can find either dark regions (cell structures)
    or bright regions (background/empty space).
    
    Args:
        img (numpy.ndarray): Input image array with values in [0,1]
        patch_size (int): Size of patches to extract
        threshold (float, optional): Intensity threshold value. Defaults to 0.2
        above_threshold (bool, optional): If True, select patches above threshold
            (empty space). If False, select patches below threshold (structures).
            Defaults to False.
        max_patches (int, optional): Maximum number of patches to return. 
            Defaults to 10000.
        scale_factor (float, optional): Factor to scale coordinates between
            preprocessed and original image. Defaults to 1.0.
        overlap (float, optional): Fraction of patch overlap during sliding.
            Must be between 0 and 1. Defaults to 0.5 (50% overlap).
            
    Returns:
        list: List of tuples containing patch coordinates (top, left) in
            original image scale
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if not 0 < overlap < 1:
        raise ValueError("overlap must be between 0 and 1")
        
    h, w = img.shape
    patch_coordinates = []
    scaled_patch_size = int(patch_size * scale_factor)
    step_size = int(scaled_patch_size * (1 - overlap))
    
    for i in range(0, h - scaled_patch_size, step_size):
        for j in range(0, w - scaled_patch_size, step_size):
            patch_xy = img[i:i+scaled_patch_size, j:j+scaled_patch_size]
            
            # Check threshold condition based on above_threshold flag
            condition = (np.all(patch_xy > threshold) if above_threshold 
                       else np.mean(patch_xy) < threshold)
            
            if condition:
                # Convert coordinates back to original image scale
                original_i = int(i / scale_factor)
                original_j = int(j / scale_factor)
                patch_coordinates.append((original_i, original_j))
                
            if len(patch_coordinates) >= max_patches:
                return patch_coordinates
                
    return patch_coordinates