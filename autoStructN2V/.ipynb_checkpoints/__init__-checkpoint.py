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

from .inference import (
    AutoStructN2VPredictor,
    visualize_denoising_result,
    compare_multiple_images
)

# Expose key utilities
from .utils.training import set_seed

# Define what gets imported with "from autoStructN2V import *"
__all__ = [
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
    
    # Inference
    'AutoStructN2VPredictor',
    'visualize_denoising_result',
    'compare_multiple_images',
    
    # Utilities
    'set_seed'
]