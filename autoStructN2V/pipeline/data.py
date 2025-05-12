# autoStructN2V/pipeline/data.py
import os
import random
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from glob import glob

from ..utils.image import get_image_paths
from ..datasets import TrainingDataset, ValidationDataset, TestDataset
from ..masking import create_stage1_mask_kernel, create_full_mask, StructuralNoiseExtractor

def split_dataset(input_dir, output_dirs, split_ratio=(0.7, 0.15, 0.15), 
                image_extension='.tif', seed=None, verbose = False):
    """
    Split dataset into training, validation, and test sets.
    
    Args:
        input_dir (str): Directory containing input images
        output_dirs (dict): Dictionary with paths to output directories
        split_ratio (tuple): Ratio for (train, val, test) split
        image_extension (str): Extension of image files
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        tuple: (train_paths, val_paths, test_paths) - Lists of image paths
    """
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
    
    # Get all images with the specified extension
    images = glob(os.path.join(input_dir, f'*{image_extension}'))
    
    if not images:
        raise ValueError(f"No images with extension {image_extension} found in {input_dir}")
    
    # Shuffle images
    random.shuffle(images)
    
    # Calculate split indices
    n_train = int(len(images) * split_ratio[0])
    n_val = int(len(images) * split_ratio[1])
    
    # Split dataset
    train_images = images[:n_train]
    val_images = images[n_train:n_train+n_val]
    test_images = images[n_train+n_val:]
    
    # Copy images to respective directories
    train_paths = _copy_images(train_images, os.path.join(output_dirs['data'], 'train'))
    val_paths = _copy_images(val_images, os.path.join(output_dirs['data'], 'val'))
    test_paths = _copy_images(test_images, os.path.join(output_dirs['data'], 'test'))
    
    if verbose:
        print("\n=== Dataset Split Details ===")
        print(f"Total images: {len(images)}")
        print(f"Training: {len(train_paths)} images ({len(train_paths)/len(images)*100:.1f}%)")
        print(f"Validation: {len(val_paths)} images ({len(val_paths)/len(images)*100:.1f}%)")
        print(f"Testing: {len(test_paths)} images ({len(test_paths)/len(images)*100:.1f}%)")
        if train_paths:
            print(f"Sample training images: {[os.path.basename(p) for p in train_paths[:3]]}")
    else:
        print(f"Dataset split: {len(train_paths)} training, {len(val_paths)} validation, {len(test_paths)} test images")
    
    
    return train_paths, val_paths, test_paths

def _copy_images(images, output_dir):
    """
    Copy images to the output directory.
    
    Args:
        images (list): List of image paths
        output_dir (str): Directory to copy images to
        
    Returns:
        list: Paths to copied images
    """
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []
    
    for img_path in images:
        filename = os.path.basename(img_path)
        dest_path = os.path.join(output_dir, filename)
        shutil.copy2(img_path, dest_path)
        output_paths.append(dest_path)
    
    return output_paths

def create_dataloaders(image_paths, config, stage="stage1", stage1_denoised_dir=None, structured_mask=None, prediction_kernel=None, verbose=False):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        image_paths (tuple): (train_paths, val_paths, test_paths)
        config (dict): Configuration dictionary
        stage (str): 'stage1' or 'stage2'
        stage1_denoised_dir (str, optional): Directory with stage1 denoised images for stage2
        structured_mask (numpy.ndarray, optional): Structured mask for stage2
        prediction_kernel (numpy.ndarray, optional): Prediction kernel for stage2
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_paths, val_paths, test_paths = image_paths
    stage_config = config[stage]
    
    # For stage2, use denoised images from stage1 if provided
    if stage == "stage2" and stage1_denoised_dir:
        # Update paths to use denoised images
        def get_denoised_paths(orig_paths, split_name):
            denoised_paths = []
            for p in orig_paths:
                base_name, ext = os.path.splitext(os.path.basename(p))
                denoised_path = os.path.join(stage1_denoised_dir, split_name, f"{base_name}_denoised{ext}")
                if os.path.exists(denoised_path):
                    denoised_paths.append(denoised_path)
                else:
                    # Try to find any matching file
                    potential_files = glob.glob(os.path.join(stage1_denoised_dir, split_name, f"{base_name}*{ext}"))
                    if potential_files:
                        denoised_paths.append(potential_files[0])
                    else:
                        print(f"Warning: No denoised file found for {p}")
            return denoised_paths
            
        train_paths = get_denoised_paths(train_paths, 'train')
        val_paths = get_denoised_paths(val_paths, 'val')
        test_paths = get_denoised_paths(test_paths, 'test')
    
    # Create stage-specific mask based on configuration
    if stage == "stage1":
        # Create a simple single-pixel mask for stage 1
        from ..masking import create_stage1_mask_kernel, create_full_mask
        single_mask = create_stage1_mask_kernel(stage_config.get('mask_center_size', 1))
        mask, prediction_kernel = create_full_mask(
            single_mask,
            stage_config['patch_size'],
            stage_config['mask_percentage'],
            verbose
        )
    else:  # stage2
        # For stage 2, use provided structured mask if available
        if structured_mask is not None and prediction_kernel is not None:
            mask = structured_mask
        else:
            # Fall back to defaults if mask not provided
            print("Warning: No structured mask provided for stage2, using defaults")
            from ..masking import create_stage1_mask_kernel, create_full_mask
            # Create a slightly larger kernel for structured behavior
            single_mask = create_stage1_mask_kernel(stage_config.get('mask_center_size', 3))
            mask, prediction_kernel = create_full_mask(
                single_mask,
                stage_config['patch_size'],
                stage_config['mask_percentage']
            )
    
    # Create datasets
    from ..datasets import TrainingDataset, ValidationDataset, TestDataset
    from torch.utils.data import DataLoader
    
    train_dataset = TrainingDataset(
        image_paths=train_paths,
        patch_size=stage_config['patch_size'],
        kernel_size=3,  # Size of neighborhood for local mean
        mask=mask,
        mask_percentage=stage_config['mask_percentage'],
        mask_strat=0,   # 0: local mean, 1: zeros, 2: random values
        prediction_kernel=prediction_kernel,
        patches_per_image=stage_config['patches_per_image'],
        use_roi=stage_config['use_roi'],
        scale_factor=stage_config['scale_factor'],
        roi_threshold=stage_config['roi_threshold'],
        select_background=stage_config['select_background'],
        use_augmentation=stage_config['use_augmentation']
    )

    val_dataset = ValidationDataset(
        image_paths=val_paths,
        patch_size=stage_config['patch_size'],
        patches_per_image=stage_config['patches_per_image'] // 2,
        use_roi=stage_config['use_roi'],
        scale_factor=stage_config['scale_factor'],
        roi_threshold=stage_config['roi_threshold'],
        select_background=stage_config['select_background']
    )

    test_dataset = TestDataset(
        image_paths=test_paths
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=stage_config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=stage_config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one full image at a time
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader