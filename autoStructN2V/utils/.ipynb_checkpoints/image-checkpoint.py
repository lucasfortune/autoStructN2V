# utils/image.py
import os
import numpy as np
from PIL import Image
from glob import glob
from skimage import exposure
from skimage.transform import resize
from skimage.filters import gaussian
from scipy.ndimage import gaussian_filter

def get_image_paths(directory):
    """
    Get all TIFF image file paths in a directory.
    
    This function searches for TIFF files (*.tif) in the specified directory. It ensures
    the directory exists and contains valid image files before returning the paths.
    
    Args:
        directory (str): Path to the directory containing TIFF images.
        
    Returns:
        list: List of absolute paths to TIFF image files.
        
    Raises:
        FileNotFoundError: If the directory doesn't exist
        ValueError: If no TIFF files are found in the directory
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
        
    image_paths = glob(os.path.join(directory, '*.tif'))
    
    if not image_paths:
        raise ValueError(f"No TIFF files found in directory: {directory}")

    folder_name = os.path.basename(os.path.normpath(directory))
    
    if os.path.exists(directory) and image_paths:
        print(f"{len(image_paths)} images found in {folder_name}.")
    return [os.path.abspath(path) for path in image_paths]

def verify_image(image_path):
    """
    Verify that a file is a valid, readable image.
    
    Args:
        image_path (str): Path to the image file to verify
        
    Returns:
        tuple: (is_valid, message) where is_valid is a boolean and message describes the result
    """
    if not os.path.exists(image_path):
        return False, "File does not exist"
    
    if not os.access(image_path, os.R_OK):
        return False, "File is not readable"
    
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify it's a valid image
        return True, "Valid image file"
    except (IOError, SyntaxError) as e:
        return False, f"Invalid image: {e}"

def load_and_normalize_image(img_path):
    """
    Load and normalize an image to floating point values between 0 and 1.
    
    This function handles various image types and bit depths:
    - 8-bit images (uint8): normalized from 0-255 to 0-1
    - 16-bit images (uint16): normalized from 0-65535 to 0-1
    - Float images: clipped to 0-1 range
    - Other types: normalized using min-max scaling
    
    Args:
        img_path (str): Path to the image file.
        
    Returns:
        numpy.ndarray: Normalized 2D image array as float32, values in [0,1]
        
    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image can't be opened or processed
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
        
    try:
        # Open the image
        with Image.open(img_path) as img:
            # Convert to numpy array
            img_array = np.array(img)

            # Handle different data types
            if img_array.dtype == np.uint8:
                # 8-bit image, already in 0-255 range
                img_normalized = img_array.astype(np.float32) / 255.0
            elif img_array.dtype == np.uint16:
                # 16-bit image, normalize from 0-65535 to 0-1
                img_normalized = img_array.astype(np.float32) / 65535.0
            elif img_array.dtype in [np.float32, np.float64]:
                # Floating point image, assume it's already normalized but clip to be safe
                img_normalized = np.clip(img_array, 0, 1).astype(np.float32)
            else:
                # For any other data type, normalize min-max to 0-1 range
                img_min, img_max = img_array.min(), img_array.max()
                img_normalized = (img_array - img_min) / (img_max - img_min)
                img_normalized = img_normalized.astype(np.float32)

            # Ensure the image is 2D (grayscale)
            if img_normalized.ndim > 2:
                img_normalized = img_normalized[:,:,0]  # Take first channel if it's somehow multi-channel
    except Exception as e:
        raise ValueError(f"Failed to process image {img_path}: {str(e)}")
    
    if img_normalized.size == 0:
        raise ValueError(f"Empty image after normalization: {img_path}")
        
    return img_normalized

def load_and_preprocess_image(img_path, sigma=300, scale_factor=0.25):
    """
    Load, preprocess, and enhance an image for training patch preselection.
    
    This function performs several steps to prepare an image for ROI detection:
    1. Loads and scales down the image to reduce computation
    2. Applies Gaussian filtering to reduce noise and smooth the image
    3. Enhances contrast using histogram equalization
    
    Args:
        img_path (str): Path to the image file
        sigma (float, optional): Standard deviation for Gaussian filter. Higher values
            create more smoothing. Scaled automatically with image size. Defaults to 300.
        scale_factor (float, optional): Factor to reduce image size. Must be between 
            0 and 1. Defaults to 0.25 (quarter size).
            
    Returns:
        numpy.ndarray: Preprocessed image array with values in [0,1]
        
    Raises:
        ValueError: If scale_factor is not between 0 and 1
        FileNotFoundError: If image file doesn't exist
    """
    # Validate inputs
    if not 0 < scale_factor <= 1:
        raise ValueError("scale_factor must be between 0 and 1")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    # Load image and ensure it's 2D
    img = Image.open(img_path)
    original_img = np.array(img)
    if original_img.ndim > 2:
        original_img = original_img[:,:,0]  # Take first channel if multi-channel

    # Scale down while preserving image characteristics
    h, w = original_img.shape
    scaled_img = resize(original_img, 
                       (int(h * scale_factor), int(w * scale_factor)),
                       anti_aliasing=True,
                       preserve_range=True).astype(np.float32)
    
    # Apply Gaussian filter with scaled sigma
    filtered_img = gaussian_filter(scaled_img, sigma=sigma * scale_factor)
    
    # Normalize and enhance contrast
    enhanced_img = exposure.equalize_hist(filtered_img)
    
    return enhanced_img