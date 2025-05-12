# inference/predictor.py
import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from ..utils.image import load_and_normalize_image
from ..utils.patching import image_to_patches, patches_to_image

class AutoStructN2VPredictor:
    """
    Predictor class for applying trained AutoStructN2V models to denoise images.
    
    This class handles loading trained models and applying them to new images
    for denoising, with support for both stages of the AutoStructN2V pipeline.
    
    Args:
        model (nn.Module): Trained denoising model
        device (torch.device, optional): Device to run inference on.
            Defaults to CUDA if available, otherwise CPU.
        patch_size (int, optional): Size of patches for processing. Defaults to 64.
        stride (int, optional): Stride for patch extraction. Defaults to patch_size//2.
    """
    def __init__(self, model, device=None, patch_size=64, stride=None):
        self.model = model
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size // 2
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model.to(self.device)
        self.model.eval()
    
    def denoise_image(self, image_path, output_path=None, show=False):
        """
        Denoise a single image using the trained model.
        
        Args:
            image_path (str): Path to the input image
            output_path (str, optional): Path to save the denoised image.
                If None, will use input_filename_denoised.tif
            show (bool, optional): Whether to display the result. Defaults to False.
            
        Returns:
            numpy.ndarray: Denoised image array
        """
        # Generate default output path if not provided
        if output_path is None:
            base_path, ext = os.path.splitext(image_path)
            output_path = f"{base_path}_denoised.tif"
        
        # Load and normalize the image
        img_array = load_and_normalize_image(image_path)

        # Calculate padding needed
        h, w = img_array.shape
        pad_h = (self.patch_size - h % self.stride) % self.stride
        pad_w = (self.patch_size - w % self.stride) % self.stride
        
        # Pad image (using reflection padding to avoid border artifacts)
        padded_img = np.pad(img_array, ((0, pad_h), (0, pad_w)), mode='reflect')
        
        # Convert to tensor (add batch and channel dimensions)
        img_tensor = torch.from_numpy(padded_img).float().unsqueeze(0).unsqueeze(0)
        
        # Denoise using patch-based approach
        denoised_tensor = self.denoise_tensor(img_tensor)
        
        # Convert back to numpy array
        denoised_array = denoised_tensor.squeeze().cpu().numpy()[:h, :w]
        
        # Save the result
        self._save_image(denoised_array, output_path)
        
        # Show results if requested
        if show:
            self._show_comparison(img_array, denoised_array)
        
        return denoised_array
    
    def denoise_tensor(self, img_tensor):
        """
        Denoise an image tensor using patch-based processing.
        
        Args:
            img_tensor (torch.Tensor): Input image tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Denoised image tensor of same shape
        """
        with torch.no_grad():
            batch_size, channels, height, width = img_tensor.shape
            
            if batch_size != 1:
                raise ValueError("This method only supports single image processing (batch_size=1)")
            
            # Process each image in the batch
            denoised_batch = []
            for b in range(batch_size):
                # Extract patches
                patches = image_to_patches(img_tensor[b], self.patch_size, self.stride)
                
                # Process patches in batches to avoid OOM
                batch_size = 16
                output_patches = []
                
                for i in range(0, len(patches), batch_size):
                    batch = patches[i:i+batch_size].to(self.device)
                    outputs = self.model(batch)
                    output_patches.append(outputs.cpu())
                
                # Concatenate batch results
                output_patches = torch.cat(output_patches, dim=0)
                
                # Reconstruct image from patches
                denoised = patches_to_image(output_patches, img_tensor[b].shape, 
                                           self.patch_size, self.stride)
                
                denoised_batch.append(denoised)
            
            # Stack the denoised images back into a batch
            return torch.stack(denoised_batch)
    
    def process_directory(self, input_dir, output_dir=None, show=False):
        """
        Process all TIFF images in a directory.
        
        Args:
            input_dir (str): Directory containing input images
            output_dir (str, optional): Directory to save output images.
                If None, will use input_dir/denoised/
            show (bool, optional): Whether to display results. Defaults to False.
            
        Returns:
            list: Paths to denoised images
        """
        # Get all TIFF files in the input directory
        image_paths = []
        for ext in ['.tif', '.tiff']:
            image_paths.extend([os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                               if f.lower().endswith(ext)])
        
        if not image_paths:
            raise ValueError(f"No TIFF images found in {input_dir}")
        
        # Create output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(input_dir, 'denoised')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each image
        output_paths = []
        for img_path in tqdm(image_paths, desc="Processing images"):
            filename = os.path.basename(img_path)
            base_name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{base_name}_denoised.tif")
            
            self.denoise_image(img_path, output_path, show=show)
            output_paths.append(output_path)
        
        return output_paths
    
    def _save_image(self, img_array, output_path):
        """
        Save an image array to disk.
        
        Args:
            img_array (numpy.ndarray): Image array to save
            output_path (str): Path to save the image
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # Normalize to 16-bit range
        img_scaled = np.clip(img_array * 65535, 0, 65535).astype(np.uint16)
        
        # Save as TIFF
        Image.fromarray(img_scaled).save(output_path)
    
    def _show_comparison(self, original, denoised):
        """
        Display original and denoised images side by side.
        
        Args:
            original (numpy.ndarray): Original image array
            denoised (numpy.ndarray): Denoised image array
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        denoised_scaled = np.clip(denoised * 65535, 0, 65535).astype(np.uint16)
        original_scaled = np.clip(original * 65535, 0, 65535).astype(np.uint16)
        
        # Display with proper normalization
        axes[0].imshow(original_scaled, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(denoised_scaled, cmap='gray')
        axes[1].set_title('Denoised')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, model_class, stage, **kwargs):
        """
        Create a predictor from a model checkpoint.
        
        Args:
            checkpoint_path (str): Path to the model checkpoint
            model_class: Model class to instantiate
            stage (str): Stage of the model ('stage1' or 'stage2')
            **kwargs: Additional arguments for predictor initialization
            
        Returns:
            AutoStructN2VPredictor: Initialized predictor with loaded model
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract hyperparameters
        if 'hparams' in checkpoint:
            hparams = checkpoint['hparams']
        else:
            hparams = {}
        
        # Create model based on stage
        if stage == 'stage1':
            features = hparams.get('n2v_features', 64)
            num_layers = hparams.get('n2v_num_layers', 4)
            patch_size = hparams.get('n2v_patch_size', 64)
        else:  # stage2
            features = hparams.get('structn2v_features', 64)
            num_layers = hparams.get('structn2v_num_layers', 4)
            patch_size = hparams.get('structn2v_patch_size', 64)
        
        # Create model
        model = model_class(features=features, num_layers=num_layers, stage=stage)
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Set patch size if not provided in kwargs
        if 'patch_size' not in kwargs:
            kwargs['patch_size'] = patch_size
        
        # Create predictor
        return cls(model=model, **kwargs)