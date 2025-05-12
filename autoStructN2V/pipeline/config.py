# autoStructN2V/pipeline/config.py
import os
import copy

def validate_config(config):
    """
    Validate and complete configuration with default values.
    
    Args:
        config (dict): User-provided configuration
        
    Returns:
        dict: Validated and completed configuration
    """
    # Create deep copy to avoid modifying original
    cfg = copy.deepcopy(config)
    
    # Set defaults for missing values
    defaults = {
        # General parameters
        'experiment_name': 'autoStructN2V_experiment',
        'output_dir': './results',
        'random_seed': 42,
        'device': 'cuda',  # 'cuda' or 'cpu'
        'split_ratio': (0.7, 0.15, 0.15),  # (train, val, test)
        'image_extension': '.tif',
        
        # Training parameters
        'num_epochs': 100,
        'early_stopping': True,
        'early_stopping_patience': 10,
        
        # First stage parameters
        'stage1': {
            'features': 64,
            'num_layers': 2,
            'patch_size': 32,
            'batch_size': 4,
            'learning_rate': 1e-4,
            'patches_per_image': 100,
            'mask_percentage': 15.0,
            'mask_center_size': 1,
            'use_roi': True,
            'roi_threshold': 0.5,
            'scale_factor': 0.25,
            'select_background': True,
            'use_augmentation': True
        },
        
        # Second stage parameters
        'stage2': {
            'features': 64,
            'num_layers': 2,
            'patch_size': 64,
            'batch_size': 2,
            'learning_rate': 1e-5,
            'patches_per_image': 200,
            'mask_percentage': 10.0,
            'use_roi': False,
            'roi_threshold': 0.5,
            'scale_factor': 0.25,
            'select_background': False,
            'use_augmentation': True,
            'extractor': {
                'norm_autocorr': True,
                'log_autocorr': True,
                'crop_autocorr': True,
                'adapt_autocorr': True,
                'adapt_CB': 50.0,
                'adapt_DF': 0.95,
                'center_size': 10,
                'base_percentile': 50,
                'percentile_decay': 1.15,
                'center_ratio_threshold': 0.3,
                'use_center_proximity': True,
                'center_proximity_threshold': 0.95,
                'keep_center_component_only': True,
                'max_true_pixels': 25
            }
        }
    }
    
    # Merge defaults with provided config
    for key, value in defaults.items():
        if key not in cfg:
            cfg[key] = value
        elif isinstance(value, dict) and isinstance(cfg[key], dict):
            # For nested dictionaries, recursively merge defaults
            for subkey, subvalue in value.items():
                if subkey not in cfg[key]:
                    cfg[key][subkey] = subvalue
    
    # Validate required parameters
    if 'input_dir' not in cfg:
        raise ValueError("input_dir must be specified in the configuration")
    
    # Ensure directories have trailing slash
    for key in ['input_dir', 'output_dir']:
        if key in cfg and not cfg[key].endswith('/'):
            cfg[key] = cfg[key] + '/'
    
    return cfg

def create_output_directories(config):
    """
    Create all necessary directories for the pipeline.
    
    Args:
        config (dict): Validated configuration
        
    Returns:
        dict: Dictionary with paths to all output directories
    """
    base_dir = config['output_dir']
    experiment_name = config['experiment_name']
    
    # Create experiment directory
    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create subdirectories
    dirs = {
        'experiment': experiment_dir,
        'data': os.path.join(experiment_dir, 'data'),
        'stage1': {
            'model': os.path.join(experiment_dir, 'stage1', 'model'),
            'logs': os.path.join(experiment_dir, 'stage1', 'logs'),
            'results': os.path.join(experiment_dir, 'stage1', 'results')
        },
        'stage2': {
            'model': os.path.join(experiment_dir, 'stage2', 'model'),
            'logs': os.path.join(experiment_dir, 'stage2', 'logs'),
            'results': os.path.join(experiment_dir, 'stage2', 'results')
        },
        'final_results': os.path.join(experiment_dir, 'final_results')
    }
    
    # Create each directory
    for _, path in dirs.items():
        if isinstance(path, dict):
            for _, subpath in path.items():
                os.makedirs(subpath, exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)
    
    # Create specific data subdirectories
    data_subdirs = ['train', 'val', 'test', 'stage1_denoised']
    for subdir in data_subdirs:
        os.makedirs(os.path.join(dirs['data'], subdir), exist_ok=True)
    
    return dirs