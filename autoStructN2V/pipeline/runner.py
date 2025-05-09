# autoStructN2V/pipeline/runner.py
import os
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
import glob

from ..utils.training import set_seed
from ..models import create_model
from ..trainers import AutoStructN2VTrainer
from ..inference import AutoStructN2VPredictor
from ..masking import StructuralNoiseExtractor, create_full_mask

from .config import validate_config, create_output_directories
from .data import split_dataset, create_dataloaders

def create_stage2_mask(denoised_patches, config):
    """
    Create a structured mask for stage 2 based on denoised patches from stage 1.
    
    Args:
        denoised_patches (numpy.ndarray): Denoised patches from stage 1
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (full_mask, prediction_kernel)
    """
    # Get extractor config, ensuring defaults if missing
    if 'stage2' not in config:
        config['stage2'] = {}
    if 'extractor' not in config['stage2']:
        config['stage2']['extractor'] = {}
    
    extractor_config = config['stage2']['extractor']
    
    # Create extractor with config parameters (using get() with defaults)
    extractor = StructuralNoiseExtractor(
        center_size=extractor_config.get('center_size', 15),
        base_percentile=extractor_config.get('base_percentile', 50),
        percentile_decay=extractor_config.get('percentile_decay', 1.1),
        center_ratio_threshold=extractor_config.get('center_ratio_threshold', 0.3),
        keep_center_component_only=extractor_config.get('keep_center_component_only', True)
    )
    
    # Extract structured mask
    struct_mask, _ = extractor.extract_mask(denoised_patches)
    
    # Create full mask and prediction kernel
    full_mask, prediction_kernel = create_full_mask(
        struct_mask,
        config['stage2'].get('patch_size', 64),
        config['stage2'].get('mask_percentage', 10.0)
    )
    
    return full_mask, prediction_kernel

def denoise_directory(model, input_dir, output_dir, config, stage):
    """
    Denoise all images in a directory.
    
    Args:
        model (nn.Module): Trained denoising model
        input_dir (str): Directory containing input images
        output_dir (str): Directory to save denoised images
        config (dict): Configuration dictionary
        stage (str): 'stage1' or 'stage2'
        
    Returns:
        list: Paths to denoised images
    """
    # Create predictor
    predictor = AutoStructN2VPredictor(
        model=model,
        patch_size=config[stage]['patch_size']
    )
    
    # Process all images
    result_paths = predictor.process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        show=False
    )
    
    return result_paths

def run_pipeline(config):
    """
    Run the complete autoStructN2V pipeline.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Summary of results
    """
    # Validate configuration
    config = validate_config(config)
    
    # Create output directories
    dirs = create_output_directories(config)
    
    # Set random seed for reproducibility
    set_seed(config['random_seed'])
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Split dataset
    print("Splitting dataset...")
    image_paths = split_dataset(
        config['input_dir'],
        dirs,
        config['split_ratio'],
        config['image_extension'],
        config['random_seed']
    )
    
    # Save a copy of the configuration
    import json
    with open(os.path.join(dirs['experiment'], 'config.json'), 'w') as f:
        # Convert any non-serializable objects to strings
        config_serializable = {k: (str(v) if not isinstance(v, (dict, list, str, int, float, bool, type(None))) else v) 
                             for k, v in config.items()}
        json.dump(config_serializable, f, indent=4)
    
    #------------------------------------------------------------------------
    # Stage 1: Standard Noise2Void
    #------------------------------------------------------------------------
    print("\n" + "="*40)
    print("Stage 1: Standard Noise2Void Training")
    print("="*40)
    
    # Create dataloaders for stage 1
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(image_paths, config, "stage1")
    
    # Create stage 1 model
    stage1_model = create_model(
        'stage1',
        features=config['stage1']['features'],
        num_layers=config['stage1']['num_layers']
    )
    
    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(stage1_model.parameters(), lr=config['stage1']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Create trainer
    stage1_trainer = AutoStructN2VTrainer(
        model=stage1_model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        hparams=config,  # Pass full config
        stage='stage1',
        experiment_name=os.path.join(dirs['stage1']['logs'], datetime.now().strftime("%Y%m%d-%H%M%S"))
    )
    
    # Train stage 1 model
    print("Training stage 1 model...")
    denoised_patches = stage1_trainer.train(train_loader, val_loader, test_loader)
    
    # Save stage 1 model
    stage1_checkpoint_path = os.path.join(dirs['stage1']['model'], 'stage1_model.pth')
    stage1_trainer.save_checkpoint(stage1_checkpoint_path)
    print(f"Saved stage 1 model to {stage1_checkpoint_path}")
    
    # Denoise all original images with stage 1 model
    print("Denoising original images with stage 1 model...")
    stage1_denoised_dir = os.path.join(dirs['data'], 'stage1_denoised')
    os.makedirs(stage1_denoised_dir, exist_ok=True)
    
    # Denoise each split separately
    for split_name, split_dir in zip(['train', 'val', 'test'], image_paths):
        split_output_dir = os.path.join(stage1_denoised_dir, split_name)
        os.makedirs(split_output_dir, exist_ok=True)
        
        # Get directory of the current split
        input_split_dir = os.path.dirname(split_dir[0])
        
        print(f"Denoising {split_name} images...")
        denoise_directory(stage1_model, input_split_dir, split_output_dir, config, 'stage1')
    
    #------------------------------------------------------------------------
    # Stage 2: Structured Noise2Void
    #------------------------------------------------------------------------
    print("\n" + "="*40)
    print("Stage 2: Structured Noise2Void Training")
    print("="*40)
    
    # Create structured mask from denoised patches
    print("Creating structured mask for stage 2...")
    stage2_mask, stage2_prediction_kernel = create_stage2_mask(denoised_patches, config)
    
    # Get paths to stage 1 denoised images
    stage1_denoised_train = []
    for p in image_paths[0]:
        base_name, ext = os.path.splitext(os.path.basename(p))
        denoised_path = os.path.join(stage1_denoised_dir, 'train', f"{base_name}_denoised{ext}")
        
        # Verify file exists
        if not os.path.exists(denoised_path):
            print(f"Warning: Expected denoised file not found: {denoised_path}")
            potential_files = glob.glob(os.path.join(stage1_denoised_dir, 'train', f"{base_name}*{ext}"))
            if potential_files:
                print(f"Found alternative file: {potential_files[0]}")
                denoised_path = potential_files[0]
            else:
                print(f"No alternative found. This may cause errors.")
        
        stage1_denoised_train.append(denoised_path)

    stage1_denoised_val = []
    for p in image_paths[1]:
        base_name, ext = os.path.splitext(os.path.basename(p))
        denoised_path = os.path.join(stage1_denoised_dir, 'val', f"{base_name}_denoised{ext}")
        if os.path.exists(denoised_path):
            stage1_denoised_val.append(denoised_path)
        else:
            print(f"Warning: Expected denoised file not found: {denoised_path}")
            potential_files = glob.glob(os.path.join(stage1_denoised_dir, 'val', f"{base_name}*{ext}"))
            if potential_files:
                stage1_denoised_val.append(potential_files[0])
    
    stage1_denoised_test = []
    for p in image_paths[2]:
        base_name, ext = os.path.splitext(os.path.basename(p))
        denoised_path = os.path.join(stage1_denoised_dir, 'test', f"{base_name}_denoised{ext}")
        if os.path.exists(denoised_path):
            stage1_denoised_test.append(denoised_path)
        else:
            print(f"Warning: Expected denoised file not found: {denoised_path}")
            potential_files = glob.glob(os.path.join(stage1_denoised_dir, 'test', f"{base_name}*{ext}"))
            if potential_files:
                stage1_denoised_test.append(potential_files[0])
    
    stage1_denoised_paths = (stage1_denoised_train, stage1_denoised_val, stage1_denoised_test)
    
    # Create custom datasets for stage 2 with the structured mask
    from ..datasets import TrainingDataset, ValidationDataset, TestDataset
    
    # Create datasets
    stage2_train_dataset = TrainingDataset(
        image_paths=stage1_denoised_train,
        patch_size=config['stage2']['patch_size'],
        kernel_size=3,
        mask=stage2_mask,
        mask_percentage=config['stage2']['mask_percentage'],
        mask_strat=0,
        prediction_kernel=stage2_prediction_kernel,
        patches_per_image=config['stage2']['patches_per_image'],
        use_roi=config['stage2']['use_roi'],
        scale_factor=config['stage2']['scale_factor'],
        roi_threshold=config['stage2']['roi_threshold'],
        select_background=config['stage2']['select_background'],
        use_augmentation=config['stage2']['use_augmentation']
    )

    stage2_val_dataset = ValidationDataset(
        image_paths=stage1_denoised_val,
        patch_size=config['stage2']['patch_size'],
        patches_per_image=config['stage2']['patches_per_image'] // 2,
        use_roi=config['stage2']['use_roi'],
        scale_factor=config['stage2']['scale_factor'],
        roi_threshold=config['stage2']['roi_threshold'],
        select_background=config['stage2']['select_background']
    )

    stage2_test_dataset = TestDataset(
        image_paths=stage1_denoised_test
    )

    # Create dataloaders for stage 2 with structured mask
    print("Creating dataloaders for stage 2...")
    stage2_train_loader, stage2_val_loader, stage2_test_loader = create_dataloaders(
        image_paths, 
        config, 
        "stage2",
        stage1_denoised_dir=stage1_denoised_dir,
        structured_mask=stage2_mask,
        prediction_kernel=stage2_prediction_kernel
    )
    
    # Create stage 2 model
    stage2_model = create_model(
        'stage2',
        features=config['stage2']['features'],
        num_layers=config['stage2']['num_layers']
    )
    
    # Create optimizer and scheduler
    stage2_optimizer = torch.optim.Adam(stage2_model.parameters(), lr=config['stage2']['learning_rate'])
    stage2_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        stage2_optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Create trainer
    stage2_trainer = AutoStructN2VTrainer(
        model=stage2_model,
        optimizer=stage2_optimizer,
        scheduler=stage2_scheduler,
        device=device,
        hparams=config,  # Pass full config
        stage='stage2',
        experiment_name=os.path.join(dirs['stage2']['logs'], datetime.now().strftime("%Y%m%d-%H%M%S"))
    )
    
    # Train stage 2 model
    print("Training stage 2 model...")
    stage2_trainer.train(stage2_train_loader, stage2_val_loader, stage2_test_loader)
    
    # Save stage 2 model
    stage2_checkpoint_path = os.path.join(dirs['stage2']['model'], 'stage2_model.pth')
    stage2_trainer.save_checkpoint(stage2_checkpoint_path)
    print(f"Saved stage 2 model to {stage2_checkpoint_path}")
    
    # Denoise stage 1 results with stage 2 model
    print("Denoising stage 1 results with stage 2 model...")
    
    # Denoise each split separately
    for split_name, input_dir in zip(['train', 'val', 'test'], 
                                    [os.path.join(stage1_denoised_dir, split) 
                                     for split in ['train', 'val', 'test']]):
        output_dir = os.path.join(dirs['final_results'], split_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Denoising {split_name} images...")
        denoise_directory(stage2_model, input_dir, output_dir, config, 'stage2')
    
    # Final summary
    print("\n" + "="*40)
    print("Pipeline Complete")
    print("="*40)
    print(f"Experiment name: {config['experiment_name']}")
    print(f"Output directory: {dirs['experiment']}")
    print(f"Stage 1 model saved to: {stage1_checkpoint_path}")
    print(f"Stage 2 model saved to: {stage2_checkpoint_path}")
    print(f"Final denoised results saved to: {dirs['final_results']}")
    
    # Return summary of results
    summary = {
        'experiment_dir': dirs['experiment'],
        'stage1_model_path': stage1_checkpoint_path,
        'stage2_model_path': stage2_checkpoint_path,
        'final_results_dir': dirs['final_results'],
        'config': config
    }
    
    return summary