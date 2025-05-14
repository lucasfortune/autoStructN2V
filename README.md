# autoStructN2V

A two-stage denoising approach combining standard and structured Noise2Void for microscopy images.

## Installation

### Setup Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/autoStructN2V.git
   cd autoStructN2V
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   # Using conda
   conda create -n autostruct python=3.8
   conda activate autostruct
   
   # Or using venv
   python -m venv autostruct_env
   source autostruct_env/bin/activate  # On Windows: autostruct_env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Install Package

1. Install in development mode:
   ```bash
   pip install -e .
   ```

2. Verify installation:
   ```python
   import autoStructN2V
   print(autoStructN2V.__version__)  # Should output '0.1.0'
   ```

## Quick Start

```python
import autoStructN2V
from autoStructN2V.pipeline import run_pipeline

# Create configuration
config = {
    'input_dir': './my_noisy_images/',
    'output_dir': './results/',
    'experiment_name': 'first_experiment',
    'device': 'cuda',  # Use 'cpu' if no GPU available
    'verbose': True    # Set to True for detailed outputs
}

# Run the complete pipeline
results = run_pipeline(config)

print(f"Pipeline completed. Final results in: {results['final_results_dir']}")
```

## Configuration Parameters

The pipeline is configured through a single dictionary with the following parameters:

### General Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|------------------|
| `input_dir` | Directory containing input images | *Required* | Valid path |
| `output_dir` | Directory to save results | './results' | Valid path |
| `experiment_name` | Name for this experiment | 'autoStructN2V_experiment' | Any string |
| `random_seed` | Seed for reproducibility | 42 | Any integer |
| `device` | Device to run training on | 'cuda' | 'cuda' or 'cpu' |
| `split_ratio` | Train/validation/test split ratio | (0.7, 0.15, 0.15) | Tuple of 3 values summing to 1 |
| `image_extension` | File extension of input images | '.tif' | '.tif', '.tiff', etc. |
| `verbose` | Whether to show detailed outputs | False | True or False |
| `num_epochs` | Maximum number of epochs | 100 | 50-200 |
| `early_stopping` | Whether to use early stopping | True | True or False |
| `early_stopping_patience` | Epochs without improvement before stopping | 10 | 5-20 |

### Stage 1 Parameters

| Parameter                  | Description                            | Default | Recommended Range  |
| -------------------------- | -------------------------------------- | ------- | ------------------ |
| `stage1/features`          | Base feature channels in UNet          | 64      | 16-128             |
| `stage1/num_layers`        | Number of UNet down/up-sampling layers | 2       | 2-4                |
| `stage1/patch_size`        | Size of image patches                  | 32      | 32-128             |
| `stage1/batch_size`        | Batch size for training                | 4       | 4-64               |
| `stage1/learning_rate`     | Learning rate                          | 1e-4    | 1e-5 to 1e-3       |
| `stage1/patches_per_image` | Number of patches extracted per image  | 100     | 50-500             |
| `stage1/mask_percentage`   | Percentage of pixels to mask           | 15.0    | 10.0-30.0          |
| `stage1/mask_center_size`  | Size of mask center                    | 1       | 1-3                |
| `stage1/use_roi`           | Whether to use ROI selection           | True    | True or False      |
| `stage1/roi_threshold`     | Threshold for ROI detection            | 0.5     | 0.3-0.7            |
| `stage1/scale_factor`      | Scale factor for ROI detection         | 0.25    | 0.1-0.5            |
| `stage1/select_background` | Whether to select background patches   | True    | True (for Stage 1) |
| `stage1/use_augmentation`  | Whether to use data augmentation       | True    | True or False      |

### Stage 2 Parameters

| Parameter                  | Description                            | Default | Recommended Range   |
| -------------------------- | -------------------------------------- | ------- | ------------------- |
| `stage2/features`          | Base feature channels in UNet          | 64      | 16-128              |
| `stage2/num_layers`        | Number of UNet down/up-sampling layers | 2       | 2-5                 |
| `stage2/patch_size`        | Size of image patches                  | 64      | 64-256              |
| `stage2/batch_size`        | Batch size for training                | 2       | 2-32                |
| `stage2/learning_rate`     | Learning rate                          | 1e-5    | 1e-6 to 1e-4        |
| `stage2/patches_per_image` | Number of patches extracted per image  | 200     | 100-1000            |
| `stage2/mask_percentage`   | Percentage of pixels to mask           | 10.0    | 5.0-20.0            |
| `stage2/use_roi`           | Whether to use ROI selection           | False   | True or False       |
| `stage2/roi_threshold`     | Threshold for ROI detection            | 0.5     | 0.3-0.7             |
| `stage2/scale_factor`      | Scale factor for ROI detection         | 0.25    | 0.1-0.5             |
| `stage2/select_background` | Whether to select background patches   | False   | False (for Stage 2) |
| `stage2/use_augmentation`  | Whether to use data augmentation       | True    | True or False       |

### Structural Noise Extractor Parameters

These parameters control how the structured noise pattern is extracted after Stage 1.

| Parameter                                     | Description                                          | Default | Recommended Range        |
| --------------------------------------------- | ---------------------------------------------------- | ------- | ------------------------ |
| `stage2/extractor/norm_autocorr`              | Whether to normalize autocorrelation                 | True    | True or False            |
| `stage2/extractor/log_autocorr`               | Whether to apply log to autocorrelation              | True    | True or False            |
| `stage2/extractor/crop_autocorr`              | Whether to crop autocorrelation to center size       | True    | True or False            |
| `stage2/extractor/adapt_autocorr`             | Whether to use adaptive thresholding                 | True    | True for real Image data |
| `stage2/extractor/adapt_CB`                   | Base coefficient for adaptive threshold              | 50.0    | 5.0-100.0                |
| `stage2/extractor/adapt_DF`                   | Distance factor for adaptive threshold               | 0.95    | 0.8-0.99                 |
| `stage2/extractor/center_size`                | Size of center square to analyze                     | 11      | 7-25                     |
| `stage2/extractor/base_percentile`            | Base percentile for thresholding                     | 50      | 30-70                    |
| `stage2/extractor/percentile_decay`           | Decay factor for threshold as rings expand           | 1.15    | 1.0-1.3                  |
| `stage2/extractor/center_ratio_threshold`     | Minimum ratio of ring max to center value            | 0.3     | 0.1-0.5                  |
| `stage2/extractor/use_center_proximity`       | Whether to use center proximity measure              | True    | True or False            |
| `stage2/extractor/center_proximity_threshold` | Threshold for center proximity                       | 0.95    | 0.5-0.99                 |
| `stage2/extractor/keep_center_component_only` | Whether to keep only connected component with center | True    | True or False            |
| `stage2/extractor/max_true_pixels`            | Maximum number of True pixels in mask                | 25      | 10-30                    |

## Advanced Configuration Example

```python
config = {
    'input_dir': './microscopy_data/',
    'output_dir': './denoised_results/',
    'experiment_name': 'my_esperiment',
    'device': 'cuda',
    'random_seed': 42,
    'verbose': True,
    'num_epochs': 150,
    'early_stopping': True,
    'early_stopping_patience': 10,
    
    'stage1': {
        'features': 64,
        'num_layers': 3,
        'patch_size': 64,
        'batch_size': 8,
        'learning_rate': 2e-4,
        'patches_per_image': 150,
        'mask_percentage': 15.0,
        'use_roi': True,
        'select_background': True
    },
    
    'stage2': {
        'features': 64,
        'num_layers': 3,
        'patch_size': 64,
        'batch_size': 4,
        'learning_rate': 1e-5,
        'patches_per_image': 300,
        'mask_percentage': 8.0,
        'use_roi': False,
        'select_background': False,
        'extractor': {
            'norm_autocorr': True,
            'log_autocorr': True,
            'crop_autocorr': True,
            'adapt_autocorr': True,
            'center_size': 11,
            'base_percentile': 60,
            'percentile_decay': 1.15,
            'use_center_proximity': True,
            'center_proximity_threshold': 0.9,
            'max_true_pixels': 20
        }
    }
}
```

## Notes on Parameter Selection

- **Patch Size**: Should be large enough to capture structural noise patterns but small enough for memory efficiency. Powers of 2 work best with UNet architecture.
- **Stage 1 vs Stage 2**: Stage 1 focuses on background noise, while Stage 2 focuses on structure-specific noise.
- **Features/Layers**: Increase for more complex noise patterns, decrease for simpler patterns or limited memory.
- **Learning Rate**: Stage 2 typically uses a lower learning rate than Stage 1.
- **Mask Percentage**: Stage 1 typically uses higher masking percentage than Stage 2.
- **Background Selection**: Set `select_background=True` for Stage 1 and `False` for Stage 2.
- **Extractor Parameters**: 
  - For structural noise with large features, increase `center_size`
  - For weak structural patterns, decrease `center_ratio_threshold`
  - To limit structural mask size, decrease `max_true_pixels`
  - To include more structural elements, increase `center_proximity_threshold`

## Troubleshooting

- **Memory Issues**: Reduce batch size, patch size, or network features. The default configuration is conservative for most systems.
- **Poor Results**: Try increasing patches_per_image, adjusting ROI parameters, or decreasing learning rate.
- **Slow Processing**: If `verbose=True` is too slow, set it to `False` for production runs.
- **Structural Noise Detection Problems**: If Stage 2 is not capturing structural noise patterns well, try adjusting extractor parameters, particularly `center_size`, `base_percentile`, and `center_ratio_threshold`.


