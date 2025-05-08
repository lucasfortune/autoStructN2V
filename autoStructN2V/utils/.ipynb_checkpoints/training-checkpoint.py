# utils/training.py
import gc
import torch
import random
import numpy as np

def validate_architecture_params(hparams):
    """
    Validate that architecture parameters work together for both stages.
    Checks minimum feature map sizes and appropriate relationships between
    stage-specific parameters.
    
    Args:
        hparams (dict): Hyperparameters dictionary
        
    Raises:
        ValueError: If architecture parameters are incompatible
    """
    # Validate first stage architecture
    n2v_min_feature_size = hparams['n2v_patch_size'] // (2 ** hparams['n2v_num_layers'])
    if n2v_min_feature_size < 4:
        raise ValueError(
            f"N2V patch size {hparams['n2v_patch_size']} is too small for "
            f"{hparams['n2v_num_layers']} layers. "
            f"Minimum feature size would be {n2v_min_feature_size}. (Reasonable would be 4)"
        )

    # Validate second stage architecture
    struct_min_feature_size = hparams['structn2v_patch_size'] // (2 ** hparams['structn2v_num_layers'])
    if struct_min_feature_size < 4:
        raise ValueError(
            f"StructN2V patch size {hparams['structn2v_patch_size']} is too small for "
            f"{hparams['structn2v_num_layers']} layers. "
            f"Minimum feature size would be {struct_min_feature_size}. (Reasonable would be 4)"
        )

    # Validate relationships between stages
    if hparams['n2v_patch_size'] > hparams['structn2v_patch_size']:
        print("Warning: First stage patch size is larger than second stage")
    
    if hparams['n2v_features'] > hparams['structn2v_features']:
        print("Warning: First stage has more features than second stage")

def get_balanced_hparams(hparams):
    """
    Adjust and validate patch numbers and other relationships between stages.
    
    Args:
        hparams (dict): Hyperparameters dictionary
        
    Returns:
        dict: hparams with any necessary adjustments
    """
    # Validate patch number ratio
    patch_ratio = hparams['structn2v_patches_per_image'] / hparams['n2v_patches_per_image']
    if patch_ratio < 2:
        print(f"Warning: Second stage might need more patches relative to first stage. "
              f"Current ratio: {patch_ratio:.2f}")

    # Validate learning rate relationship
    if hparams['structn2v_learning_rate'] > hparams['n2v_learning_rate']:
        print("Warning: Second stage learning rate is higher than first stage")

    # Validate batch size relationship
    if hparams['structn2v_batch_size'] > hparams['n2v_batch_size']:
        print("Warning: Second stage batch size is larger than first stage")

    return hparams

def estimate_memory_requirements(hparams):
    """
    Estimate memory requirements for training both stages.
    
    Args:
        hparams (dict): Hyperparameters dictionary
        
    Returns:
        float: Maximum memory requirement in MB between the two stages
    """
    # First stage memory estimation
    n2v_patch_memory = hparams['n2v_patch_size'] * hparams['n2v_patch_size'] * 4
    n2v_batch_memory = n2v_patch_memory * hparams['n2v_batch_size']
    n2v_model_memory = (hparams['n2v_features'] * (2 ** hparams['n2v_num_layers']) * 
                       n2v_patch_memory // 64)
    n2v_total_memory = (n2v_batch_memory + n2v_model_memory) * 3

    # Second stage memory estimation
    struct_patch_memory = hparams['structn2v_patch_size'] * hparams['structn2v_patch_size'] * 4
    struct_batch_memory = struct_patch_memory * hparams['structn2v_batch_size']
    struct_model_memory = (hparams['structn2v_features'] * (2 ** hparams['structn2v_num_layers']) * 
                          struct_patch_memory // 64)
    struct_total_memory = (struct_batch_memory + struct_model_memory) * 3

    # Return maximum memory requirement
    return max(n2v_total_memory, struct_total_memory) / (1024 * 1024)

def cleanup(doc: bool = False) -> None:
    """
    Clean up PyTorch objects and free memory/GPU resources.
    
    This function attempts to delete specific PyTorch objects and runs garbage collection
    to free up memory. It handles both CPU and GPU memory cleanup.
    
    Args:
        doc (bool, optional): If True, print status messages for each cleanup operation.
            Defaults to False.
    """
    objects_to_cleanup = [
        'n2v_train_loader', 'structn2v_train_loader', 'val_loader', 'test_loader',
        'structn2v_model', 'n2v_model', 'optimizer', 'scheduler',
        'all_correlations', 'inverted_arrays'
    ]
    
    for obj_name in objects_to_cleanup:
        try:
            exec(f"del {obj_name}")
            if doc:
                print(f'{obj_name} deleted')
        except NameError:
            if doc:
                print(f'{obj_name} not found')
    
    # Clean up memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all random number generators.
    
    This function ensures deterministic behavior by setting seeds for:
    - Python's random module
    - NumPy's random number generator
    - PyTorch's random number generator
    - CUDA's random number generator
    - cuDNN's deterministic mode
    
    Args:
        seed (int): The random seed to use
        
    Note:
        Setting deterministic mode may impact performance as some optimized
        algorithms will be disabled.
    """
    if not isinstance(seed, int):
        raise TypeError("Seed must be an integer")
        
    # Set Python's random seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set PyTorch's random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        
    # Enable deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # for reproducibility