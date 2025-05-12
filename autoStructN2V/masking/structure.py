# autoStructN2V/masking/structure.py
import numpy as np
from scipy import signal
from skimage.measure import label

class StructuralNoiseExtractor:
    """
    A class for extracting structural binary masks from noisy images using autocorrelation analysis.
    
    This class can process either a single image or multiple images by averaging their autocorrelations.
    It implements a ring-based approach to identify structural patterns in noise.
    """
    
    def __init__(self, 
                 norm_autocorr=True,
                 log_autocorr=True,
                 crop_autocorr=True,
                 adapt_autocorr=True,
                 adapt_CB=50.0,
                 adapt_DF=0.95,
                 center_size=15,
                 base_percentile=50,
                 percentile_decay=1.15,
                 center_ratio_threshold=0.3,
                 use_center_proximity=False,
                 center_proximity_threshold=0.8,
                 keep_center_component_only=True,
                 max_true_pixels=None):
        """
        Initialize the structural noise extractor with configuration parameters.
        
        Args:
            norm_autocorr (bool): Whether to normalize the autocorrelation
            log_autocorr (bool): Whether to apply log to autocorrelation values
            crop_autocorr (bool): Whether to crop to center_size
            adapt_autocorr (bool): Whether to use adaptive thresholding
            adapt_CB (float): Base coefficient for adaptive threshold
            adapt_DF (float): Distance factor for adaptive threshold
            center_size (int): Size of center square to analyze
            base_percentile (float): Base percentile for thresholding
            percentile_decay (float): Decay factor for threshold as rings expand
            center_ratio_threshold (float): Minimum ratio of ring max to center value
            use_center_proximity (bool): Whether to use center proximity measure
            center_proximity_threshold (float): Threshold for center proximity
            keep_center_component_only (bool): Whether to keep only connected component with center
            max_true_pixels (int): Maximum number of True pixels in mask
        """
        # Store configuration parameters
        self.norm_autocorr = norm_autocorr
        self.log_autocorr = log_autocorr
        self.crop_autocorr = crop_autocorr
        self.adapt_autocorr = adapt_autocorr
        self.adapt_CB = adapt_CB
        self.adapt_DF = adapt_DF
        self.center_size = center_size
        self.base_percentile = base_percentile
        self.percentile_decay = percentile_decay
        self.center_ratio_threshold = center_ratio_threshold
        self.use_center_proximity = use_center_proximity
        self.center_proximity_threshold = center_proximity_threshold
        self.keep_center_component_only = keep_center_component_only
        self.max_true_pixels = max_true_pixels
        
        # Initialize cache for intermediate results
        self._autocorr = None
        self._center_square = None
        self._ring_coordinates = None
        self._center_value = None
        self._original_autocorr_values = None
    
    def _calculate_autocorrelation(self, img):
        """
        Calculate the autocorrelation of an image.
        
        Args:
            img (numpy.array): Noisy image (2D array)
            
        Returns:
            numpy.array: Autocorrelation
        """
        # Convert to numpy if tensor
        if hasattr(img, 'numpy'):
            img = img.numpy()
            
        # Handle different shapes
        if len(img.shape) == 3:
            # (batch, height, width)
            if img.shape[0] == 1:
                img = img[0]
            else:
                # Average across batch dimension
                img = np.mean(img, axis=0)
        elif len(img.shape) == 4:
            # (batch, channels, height, width)
            if img.shape[0] == 1 and img.shape[1] == 1:
                img = img[0, 0]
            else:
                # Average across batch dimension and use first channel
                img = np.mean(img, axis=0)[0]
                
        # Ensure it's 2D
        if len(img.shape) != 2:
            raise ValueError(f"Could not convert input shape {img.shape} to 2D image")
        
        # Center the noise pattern
        noise_centered = img - np.mean(img)
        # Calculate autocorrelation
        autocorr = signal.fftconvolve(noise_centered, np.flip(np.flip(noise_centered, 0), 1), mode='full')
        
        if self.norm_autocorr:
            center_y, center_x = autocorr.shape[0]//2, autocorr.shape[1]//2
            autocorr = autocorr / (autocorr[center_y, center_x] + 1e-10)
        
        if self.log_autocorr:
            autocorr = np.log1p(np.abs(autocorr))
        
        return autocorr
    
    def _crop_and_adapt(self, autocorr):
        """
        Crop the central region of the autocorrelation and adapt it with threshold matrix.
        
        Args:
            autocorr (numpy.ndarray): Autocorrelation array
            
        Returns:
            numpy.ndarray: Processed center square
        """
        # Extract center square
        if self.crop_autocorr:
            margin = self.center_size // 2
            center_y, center_x = autocorr.shape[0]//2, autocorr.shape[1]//2
            center_square = autocorr[center_y-margin:center_y+margin+1, 
                                   center_x-margin:center_x+margin+1]
        else:
            margin = autocorr.shape[1]//2
            center_square = autocorr
        
        # Create Threshold Matrix
        if self.adapt_autocorr:
            # Create distance matrix from center
            y, x = np.ogrid[-margin:margin+1, -margin:margin+1]
            distance = np.sqrt(x*x + y*y)
            
            # Normalize distance to 0-1 range
            max_distance = np.sqrt(2) * margin
            normalized_distance = distance / max_distance
            
            # Create radially-adaptive threshold matrix
            base_threshold = np.median(center_square)
            threshold_multiplier = self.adapt_CB * np.power(self.adapt_DF, normalized_distance * 10)
            threshold_matrix = base_threshold * threshold_multiplier
            
            # Normalize center_square and threshold_matrix
            center_square_norm = center_square - np.min(center_square)
            center_square = center_square_norm / np.max(center_square_norm)
            
            threshold_matrix_norm = threshold_matrix - np.min(threshold_matrix)
            threshold_matrix = (threshold_matrix_norm / np.max(threshold_matrix_norm) / 2)
            
            # Adapt normalized cropped autocorrelation
            center_square = center_square - threshold_matrix
        
        return center_square
    
    def _create_ring_coordinates(self, center_y, center_x, max_rings):
        """
        Pre-compute coordinates for all rings.
        
        Args:
            center_y (int): Y coordinate of center
            center_x (int): X coordinate of center
            max_rings (int): Maximum number of rings to create
            
        Returns:
            list: List of coordinate lists for each ring
        """
        ring_coordinates = []
        center_size = self._center_square.shape[0]
        
        for ring_idx in range(1, max_rings):
            inner_size = 2 * ring_idx - 1
            outer_size = 2 * ring_idx + 1
            
            if outer_size > center_size:
                outer_size = center_size
                if inner_size >= outer_size:
                    break
            
            # Calculate boundaries
            inner_start = center_y - (inner_size // 2)
            inner_end = inner_start + inner_size
            outer_start = center_y - (outer_size // 2)
            outer_end = outer_start + outer_size
            
            # Get coordinates for this ring
            coords = []
            
            # Top and bottom rows
            for i in [outer_start, outer_end - 1]:
                if 0 <= i < center_size:
                    for j in range(outer_start, outer_end):
                        if 0 <= j < center_size:
                            coords.append((i, j))
            
            # Left and right columns (excluding corners)
            for j in [outer_start, outer_end - 1]:
                if 0 <= j < center_size:
                    for i in range(outer_start + 1, outer_end - 1):
                        if 0 <= i < center_size:
                            coords.append((i, j))
            
            ring_coordinates.append(coords)
        
        return ring_coordinates
    
    def _extract_ring_pixels(self, ring_coords):
        """
        Extract pixel values from a ring using coordinates.
        
        Args:
            ring_coords (list): List of (y, x) coordinates
            
        Returns:
            numpy.ndarray: Values of pixels at coordinates
        """
        if not ring_coords:
            return np.array([])
        
        # Convert coordinates to arrays for efficient indexing
        rows, cols = zip(*ring_coords)
        return self._center_square[rows, cols]
    
    def _process_ring(self, ring_coords, ring_idx):
        """
        Process a single ring and return the threshold and valid pixel mask.
        
        Args:
            ring_coords (list): List of (y, x) coordinates for the ring
            ring_idx (int): Index of current ring (1-based)
            
        Returns:
            tuple: (threshold, should_process, above_threshold_mask)
        """
        ring_pixels = self._extract_ring_pixels(ring_coords)
        
        if len(ring_pixels) == 0:
            return 0.0, False, np.array([], dtype=bool)
        
        # Calculate percentile threshold
        current_percentile = self.base_percentile * (self.percentile_decay ** (ring_idx - 1))
        
        # Calculate threshold
        min_val = np.min(ring_pixels)
        max_val = np.max(ring_pixels)
        value_range = max_val - min_val
        threshold = min_val + (value_range * current_percentile / 100)
        
        # Check if ring contains significant structure
        print(f"ring{ring_idx}: max_value: {max_val}")
        should_process = max_val >= self._center_value * self.center_ratio_threshold
        
        # Create mask for pixels above threshold
        if should_process:
            # Traditional threshold check
            above_threshold_mask = ring_pixels > threshold
            
            # Add proximity check if enabled
            if self.use_center_proximity:
                # Check if pixel value / center value > proximity threshold
                proximity_mask = (ring_pixels / self._center_value) > self.center_proximity_threshold
                
                # Combine both conditions with OR: True if either condition is met
                above_threshold_mask = above_threshold_mask | proximity_mask
                
                # Debug logging
                if np.any(proximity_mask & ~(ring_pixels > threshold)):
                    added_by_proximity = np.sum(proximity_mask & ~(ring_pixels > threshold))
                    print(f"ring{ring_idx}: {added_by_proximity} additional pixels added by proximity check")
        else:
            above_threshold_mask = np.array([], dtype=bool)
        
        return threshold, should_process, above_threshold_mask
    
    def _limit_mask_size(self, binary_mask):
        """
        Limit the number of True pixels in the mask to max_true_pixels,
        keeping only pixels with the highest original autocorrelation values.
        
        Args:
            binary_mask (numpy.ndarray): Binary mask
            
        Returns:
            numpy.ndarray: Limited binary mask
        """
        if self.max_true_pixels is None:
            return binary_mask  # No limitation requested
            
        # Count current True pixels
        true_count = np.sum(binary_mask)
        
        # If already below threshold, return as is
        if true_count <= self.max_true_pixels:
            return binary_mask
            
        # Get coordinates of all True pixels
        true_coords = np.where(binary_mask)
        
        # Always ensure center pixel is kept
        center_y, center_x = binary_mask.shape[0] // 2, binary_mask.shape[1] // 2
        
        # Get original autocorrelation values for these pixels
        values = self._original_autocorr_values[true_coords]
        
        # Sort coordinates by autocorrelation value (descending)
        sort_indices = np.argsort(-values)  # Negative for descending order
        
        # Create a new mask with only the top max_true_pixels
        limited_mask = np.zeros_like(binary_mask)
        
        # Always keep center pixel
        limited_mask[center_y, center_x] = True
        
        # Keep track of how many pixels we've added
        pixels_added = 1  # Already added center
        
        # Add the highest-value pixels
        for idx in sort_indices:
            y, x = true_coords[0][idx], true_coords[1][idx]
            
            # Skip center pixel (already added)
            if y == center_y and x == center_x:
                continue
                
            limited_mask[y, x] = True
            pixels_added += 1
            
            # Stop when we reach the desired limit
            if pixels_added >= self.max_true_pixels:
                break
        
        # Report how many pixels were removed
        removed_count = true_count - pixels_added
        if removed_count > 0:
            print(f"Limited mask size: removed {removed_count} pixels, kept {pixels_added} pixels with highest values")
            
        return limited_mask
    
    def extract_mask(self, noise_patterns):
        """
        Extract binary mask from a list of noisy patterns by averaging their autocorrelations.
        
        Args:
            noise_patterns (list or numpy.ndarray): List of noisy images or single noisy image
                
        Returns:
            tuple: (binary_mask, center_square)
                - binary_mask (numpy.ndarray): Binary mask of structural noise
                - center_square (numpy.ndarray): Processed autocorrelation center
        """
        # Reset cache for new processing
        self.clear_cache()
        
        # Convert tensor to numpy if needed
        if hasattr(noise_patterns, 'numpy'):
            noise_patterns = noise_patterns.numpy()
        
        # Handle different input types
        if isinstance(noise_patterns, list):
            # List of images - compute autocorrelation for each
            autocorrs = [self._calculate_autocorrelation(pattern) for pattern in noise_patterns]
        elif isinstance(noise_patterns, np.ndarray):
            if len(noise_patterns.shape) <= 2:
                # Single 2D image
                self._autocorr = self._calculate_autocorrelation(noise_patterns)
                autocorrs = None
            else:
                # 3D+ array - treat as batch of images
                autocorrs = []
                # Process each image in the batch
                for i in range(noise_patterns.shape[0]):
                    if len(noise_patterns.shape) == 3:  # (batch, h, w)
                        pattern = noise_patterns[i]
                    elif len(noise_patterns.shape) == 4:  # (batch, c, h, w)
                        pattern = noise_patterns[i, 0]  # Take first channel
                    else:
                        raise ValueError(f"Cannot process noise patterns with shape {noise_patterns.shape}")
                    
                    autocorr = self._calculate_autocorrelation(pattern)
                    autocorrs.append(autocorr)
        else:
            raise TypeError(f"Expected list or numpy.ndarray, got {type(noise_patterns)}")
        
        # Average autocorrelations if we have multiple
        if autocorrs:
            # Make sure all autocorrelations have the same shape
            shapes = [a.shape for a in autocorrs]
            if len(set(shapes)) > 1:
                # Find minimum dimensions to crop all to same size
                min_h = min(s[0] for s in shapes)
                min_w = min(s[1] for s in shapes)
                
                # Crop all autocorrelations to same size
                cropped_autocorrs = []
                for a in autocorrs:
                    center_y, center_x = a.shape[0]//2, a.shape[1]//2
                    h_start = center_y - min_h//2
                    h_end = h_start + min_h
                    w_start = center_x - min_w//2
                    w_end = w_start + min_w
                    cropped_autocorrs.append(a[h_start:h_end, w_start:w_end])
                
                self._autocorr = np.mean(cropped_autocorrs, axis=0)
            else:
                self._autocorr = np.mean(autocorrs, axis=0)
                
            print(f"Averaged autocorrelations from {len(autocorrs)} images")
            
        # Process center square and cache it
        self._center_square = self._crop_and_adapt(self._autocorr)
        
        # Store the original center square values before any processing
        self._original_autocorr_values = self._center_square.copy()
        
        center_size = self._center_square.shape[0]
        
        # Initialize binary mask
        binary_mask = np.zeros_like(self._center_square, dtype=bool)
        
        # Set center pixel and get its value
        center_y, center_x = center_size // 2, center_size // 2
        binary_mask[center_y, center_x] = True
        self._center_value = self._center_square[center_y, center_x]
        print(f"center value: {self._center_value}")
        
        # Pre-compute all ring coordinates
        max_rings = max(center_y, center_x) + 1
        self._ring_coordinates = self._create_ring_coordinates(center_y, center_x, max_rings)
        
        # Process each ring
        for ring_idx, ring_coords in enumerate(self._ring_coordinates, start=1):
            # Process the ring and get threshold and mask
            threshold, should_process, above_threshold_mask = self._process_ring(ring_coords, ring_idx)
            
            # Skip ring if it doesn't contain significant structure
            if not should_process:
                continue
            
            # Apply threshold to ring pixels efficiently
            if len(above_threshold_mask) > 0:
                # Convert ring coordinates to boolean mask efficiently
                ring_y, ring_x = zip(*ring_coords)
                binary_mask[ring_y, ring_x] = above_threshold_mask
        
        # Post-processing: Keep only the connected component containing the center pixel
        if self.keep_center_component_only:
            # Label connected components
            labeled_mask = label(binary_mask, connectivity=2)  # 8-connectivity
            
            # Find the label of the component containing the center pixel
            center_label = labeled_mask[center_y, center_x]
            
            # Create new mask with only the center component
            if center_label > 0:  # center_label is 0 if center is not True (shouldn't happen)
                binary_mask = (labeled_mask == center_label)
                
                # Count removed components for debugging
                num_components = np.max(labeled_mask)
                if num_components > 1:
                    print(f"Removed {num_components - 1} disconnected components, kept component {center_label}")
            else:
                print("Warning: Center pixel was not True in the binary mask!")
        
        # Apply size limit if requested
        binary_mask = self._limit_mask_size(binary_mask)
        
        return binary_mask, self._center_square
    
    def clear_cache(self):
        """
        Clear cached results for new processing.
        """
        self._autocorr = None
        self._center_square = None
        self._ring_coordinates = None
        self._center_value = None
        self._original_autocorr_values = None