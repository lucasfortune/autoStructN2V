# autoStructN2V/__init__.py

# Version information
__version__ = '0.1.0'

# Import key components from subpackages
from .datasets import (
    BaseNoiseDataset,
    TrainingDataset,
    ValidationDataset,
    TestDataset
)

from .models import (
    FlexibleUNet,
    AutoStructN2VModel,
    create_model
)

from .trainers import (
    AutoStructN2VTrainer,
    EarlyStopping
)

from .masking import (
    create_stage1_mask_kernel,
    StructuralNoiseExtractor,
    create_full_mask,
    create_mask_for_training
)

from .inference import (
    AutoStructN2VPredictor,
    visualize_denoising_result,
    compare_multiple_images
)

# Expose key utilities
from .utils.training import set_seed

from .pipeline import run_pipeline

# Define what gets imported with "from autoStructN2V import *"
__all__ = [

    'run_pipeline',
    
    # Datasets
    'BaseNoiseDataset',
    'TrainingDataset',
    'ValidationDataset',
    'TestDataset',
    
    # Models
    'FlexibleUNet',
    'AutoStructN2VModel',
    'create_model',
    
    # Trainers
    'AutoStructN2VTrainer',
    'EarlyStopping',

       # Masking
    'create_stage1_mask_kernel',
    'StructuralNoiseExtractor',
    'create_full_mask',
    'create_mask_for_training',
    
    # Inference
    'AutoStructN2VPredictor',
    'visualize_denoising_result',
    'compare_multiple_images',
    
    # Utilities
    'set_seed'
]