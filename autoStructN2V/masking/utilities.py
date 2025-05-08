# autoStructN2V/masking/utilities.py
import numpy as np

def create_full_mask(single_masking_kernel, patch_size, mask_percentage):
    """
    Create a numpy array with efficient random placements of the input square pattern.
    
    Args:
        single_masking_kernel (numpy.ndarray): kernel representing a single mask 
                            (single True with False border for Stage 1, more complex for Stage 2)
        patch_size (int): Size of the full mask (same size as patch to denoise)
        mask_percentage (float): Target percentage of True values in the output (0-100)
    
    Returns:
        tuple: (full_masking_kernel, prediction_kernel)
            - full_masking_kernel: Boolean array with random placements of the input single mask pattern
            - prediction_kernel: Boolean array with only the center points of each pattern marked as True
    """
    # Create empty output arrays
    full_masking_kernel = np.zeros((patch_size, patch_size), dtype=bool)
    prediction_kernel = np.zeros((patch_size, patch_size), dtype=bool)
    
    # Get dimensions and properties of the input pattern
    pattern_size = single_masking_kernel.shape[0]
    true_count_per_pattern = np.sum(single_masking_kernel)
    
    # Calculate total number of pixels and target number of True pixels
    total_pixels = patch_size * patch_size
    target_true_pixels = int(total_pixels * (mask_percentage / 100))
    
    # Calculate how many patterns we need to place (rounded up to handle edge cases better)
    num_patterns_needed = int(np.ceil(target_true_pixels / true_count_per_pattern))
    
    # Create a grid of all possible positions
    y, x = np.meshgrid(
        np.arange(patch_size - pattern_size + 1),
        np.arange(patch_size - pattern_size + 1)
    )
    positions = np.column_stack((x.ravel(), y.ravel()))
    
    # Shuffle positions for randomness
    np.random.shuffle(positions)
    
    # Find where the True values are in the pattern relative to the origin
    pattern_true_coords = np.argwhere(single_masking_kernel)
    
    # Find the center point of the pattern
    center_y, center_x = pattern_size // 2, pattern_size // 2
    
    # Track how many patterns we've placed
    patterns_placed = 0
    
    # Try to place patterns
    for pos_idx, (i, j) in enumerate(positions):
        if patterns_placed >= num_patterns_needed:
            break
            
        # Check if pattern would fit without adjacent True values
        is_valid = True
        
        # Calculate where the True values would be if placed at this position
        true_positions = pattern_true_coords + [i, j]
        
        # Check if any True value would touch an existing True value
        for y, x in true_positions:
            # Skip if out of bounds
            if y < 0 or y >= patch_size or x < 0 or x >= patch_size:
                is_valid = False
                break
                
            # Check all 8 adjacent cells + the cell itself
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < patch_size and 0 <= nx < patch_size:
                        if full_masking_kernel[ny, nx] and not (dy == 0 and dx == 0 and single_masking_kernel[y-i, x-j]):
                            is_valid = False
                            break
                if not is_valid:
                    break
            if not is_valid:
                break
        
        if is_valid:
            # Place the pattern
            for y, x in true_positions:
                if 0 <= y < patch_size and 0 <= x < patch_size:
                    full_masking_kernel[y, x] = True
            
            # Mark the center point in the second mask
            center_y_pos = i + center_y
            center_x_pos = j + center_x
            if 0 <= center_y_pos < patch_size and 0 <= center_x_pos < patch_size:
                prediction_kernel[center_y_pos, center_x_pos] = True
            
            patterns_placed += 1
            
            # Optimization: Remove nearby positions from consideration
            if pos_idx < len(positions) - 1:
                distances = np.abs(positions[pos_idx+1:] - np.array([i, j]))
                keep_mask = np.any(distances >= pattern_size - 1, axis=1)
                positions = np.vstack([positions[:pos_idx+1], positions[pos_idx+1:][keep_mask]])
    
    # Calculate the actual percentage achieved
    actual_percentage = (np.sum(full_masking_kernel) / total_pixels) * 100
    
    # If we haven't reached our target, try to add individual patterns to get closer
    if actual_percentage < mask_percentage * 0.9 and patterns_placed > 0:
        # Find empty spaces where we might place additional patterns
        remaining_space = ~full_masking_kernel
        
        # Try to place more patterns in empty regions
        for i in range(0, patch_size - pattern_size + 1, pattern_size - 1):
            for j in range(0, patch_size - pattern_size + 1, pattern_size - 1):
                if np.sum(full_masking_kernel) / total_pixels >= mask_percentage / 100:
                    break
                    
                region = remaining_space[i:i+pattern_size, j:j+pattern_size]
                if region.shape == single_masking_kernel.shape and np.all(region):
                    # Place pattern
                    full_masking_kernel[i:i+pattern_size, j:j+pattern_size] = single_masking_kernel | full_masking_kernel[i:i+pattern_size, j:j+pattern_size]
                    
                    # Mark the center point in the second mask
                    center_y_pos = i + center_y
                    center_x_pos = j + center_x
                    if 0 <= center_y_pos < patch_size and 0 <= center_x_pos < patch_size:
                        prediction_kernel[center_y_pos, center_x_pos] = True
    
    # Final actual percentage
    actual_percentage = (np.sum(full_masking_kernel) / total_pixels) * 100
    print(f"Achieved {actual_percentage:.2f}% True values (target: {mask_percentage}%)")
    print(f"Placed {np.sum(prediction_kernel)} pattern centers")
    
    return full_masking_kernel, prediction_kernel

def create_mask_for_training(stage, kernel=None, patch_size=64, mask_percentage=20.0, **kwargs):
    """
    Create a mask for training stage 1 or stage 2.
    
    This is a convenience function that creates appropriate masks for each stage.
    
    Args:
        stage (str): 'stage1' for standard N2V or 'stage2' for structured N2V
        kernel (numpy.ndarray, optional): Custom kernel to use. If None, creates appropriate kernel.
        patch_size (int): Size of patches to be processed
        mask_percentage (float): Percentage of pixels to mask (0-100)
        **kwargs: Additional parameters for kernel generation
        
    Returns:
        tuple: (full_mask, prediction_kernel)
            - full_mask: Boolean array with masks applied
            - prediction_kernel: Corresponding prediction kernel for loss calculation
    """
    from .kernels import create_stage1_mask_kernel
    
    if stage not in ['stage1', 'stage2']:
        raise ValueError("Stage must be either 'stage1' or 'stage2'")
    
    # Create or validate kernel
    if kernel is None:
        if stage == 'stage1':
            # For stage1, create a simple single-pixel mask
            center_size = kwargs.get('center_size', 1)
            kernel = create_stage1_mask_kernel(center_size)
        else:  # stage2
            # For stage2, require a pre-created structured mask from StructuralNoiseExtractor
            raise ValueError("For stage2, you must provide a structured kernel. Use StructuralNoiseExtractor to create one.")
    
    # Create full mask from kernel
    full_mask, prediction_kernel = create_full_mask(
        kernel, 
        patch_size, 
        mask_percentage
    )
    
    return full_mask, prediction_kernel