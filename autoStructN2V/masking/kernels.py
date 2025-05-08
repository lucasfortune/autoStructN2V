# autoStructN2V/masking/kernels.py
import numpy as np

def create_stage1_mask_kernel(center_size):
    """
    Create a square numpy array of type bool with a square of True values 
    in the center surrounded by a 1-cell wide border of False values.
    
    Args:
        center_size (int): Size of the center square of True values
    
    Returns:
        single_stage1_masking_kernel (np.array): A square boolean array with a True center and False border
    """
    if center_size < 1:
        raise ValueError("center_size must be at least 1")
    
    # The full array size is the center size plus 2 (for the borders)
    full_size = center_size + 2

    if full_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    
    # Create an array of False values
    single_stage1_masking_kernel = np.zeros((full_size, full_size), dtype=bool)
    
    # Set the center square to True
    single_stage1_masking_kernel[1:-1, 1:-1] = True
    
    return single_stage1_masking_kernel

def create_blind_spot_kernel(kernel_size=3):
    """
    Create a typical blind-spot kernel used in Noise2Void.
    This is a 2D array with True values everywhere except the center.
    
    Args:
        kernel_size (int): Size of the kernel (must be odd)
        
    Returns:
        numpy.ndarray: Boolean kernel with False only at the center
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    
    kernel = np.ones((kernel_size, kernel_size), dtype=bool)
    center = kernel_size // 2
    kernel[center, center] = False
    
    return kernel