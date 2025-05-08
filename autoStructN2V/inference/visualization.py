# inference/visualization.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from skimage.exposure import equalize_hist

def visualize_denoising_result(original, denoised, title=None, figsize=(12, 6), cmap='gray'):
    """
    Visualize original and denoised images side by side with histograms.
    
    Args:
        original (numpy.ndarray): Original image array
        denoised (numpy.ndarray): Denoised image array
        title (str, optional): Overall figure title. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to (12, 6).
        cmap (str, optional): Colormap to use. Defaults to 'gray'.
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Ensure same normalization for both images
    vmin = min(original.min(), denoised.min())
    vmax = max(original.max(), denoised.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Display original image
    im0 = axes[0, 0].imshow(original, cmap=cmap, norm=norm)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Display denoised image
    im1 = axes[0, 1].imshow(denoised, cmap=cmap, norm=norm)
    axes[0, 1].set_title('Denoised')
    axes[0, 1].axis('off')
    
    # Add colorbar
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Display histograms
    axes[1, 0].hist(original.flatten(), bins=100, alpha=0.7)
    axes[1, 0].set_title('Original Histogram')
    axes[1, 0].set_xlim(vmin, vmax)
    
    axes[1, 1].hist(denoised.flatten(), bins=100, alpha=0.7)
    axes[1, 1].set_title('Denoised Histogram')
    axes[1, 1].set_xlim(vmin, vmax)
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    return fig

def compare_multiple_images(image_arrays, titles=None, figsize=(15, 10), cmap='gray'):
    """
    Compare multiple images in a grid layout.
    
    Args:
        image_arrays (list): List of image arrays to compare
        titles (list, optional): List of titles for each image. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to (15, 10).
        cmap (str, optional): Colormap to use. Defaults to 'gray'.
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    n_images = len(image_arrays)
    
    # Calculate grid dimensions
    cols = min(n_images, 3)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle single row/column case
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(-1)
    
    # Find global min/max for consistent normalization
    vmin = min(img.min() for img in image_arrays)
    vmax = max(img.max() for img in image_arrays)
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Plot each image
    for i, img in enumerate(image_arrays):
        if i < rows * cols:
            ax = axes.flat[i]
            im = ax.imshow(img, cmap=cmap, norm=norm)
            
            if titles and i < len(titles):
                ax.set_title(titles[i])
            
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for i in range(n_images, rows * cols):
        axes.flat[i].axis('off')
    
    plt.tight_layout()
    return fig

def create_difference_map(original, denoised, enhanced=True):
    """
    Create a difference map between original and denoised images.
    
    Args:
        original (numpy.ndarray): Original image array
        denoised (numpy.ndarray): Denoised image array
        enhanced (bool, optional): Whether to enhance contrast of the difference map.
            Defaults to True.
            
    Returns:
        numpy.ndarray: Difference map
    """
    # Calculate absolute difference
    diff = np.abs(original - denoised)
    
    # Enhance contrast if requested
    if enhanced:
        diff = equalize_hist(diff)
    
    return diff