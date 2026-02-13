"""
Base preprocessor interface for image preprocessing.

This module defines the abstract base class that all preprocessors must inherit from.
Custom preprocessors must use JAX for high-performance preprocessing.
"""

from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Optional
import numpy as np
import jax
import jax.numpy as jnp
from loguru import logger


class BaseJAXPreprocessor(ABC):
    """
    Abstract base class for JAX-based image preprocessors.
    
    All custom preprocessors must inherit from this class and use JAX
    for preprocessing operations. This ensures high performance through
    JIT compilation and vectorization.
    
    Subclasses must implement:
        - _preprocess_single_jax(): JAX-based single image preprocessing logic
    
    The base class provides:
        - JAX device configuration (CPU/GPU/TPU)
        - Device placement utilities
        - Image loading from files/URLs
        - Batch processing with vmap
        - Standard preprocessing pipeline
    
    Example:
        ```python
        from src.base_preprocessor import BaseJAXPreprocessor
        import jax.numpy as jnp
        
        class MyCustomPreprocessor(BaseJAXPreprocessor):
            def __init__(self, image_size=(224, 224), **kwargs):
                super().__init__(image_size=image_size, **kwargs)
                # Your custom initialization
            
            def _preprocess_single_jax(self, image: jnp.ndarray) -> jnp.ndarray:
                # Your custom JAX preprocessing logic
                # Use JAX operations for best performance
                resized = jax.image.resize(image, (*self.image_size, 3), method='bilinear')
                normalized = resized / 255.0
                return normalized
        ```
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        use_gpu: bool = False,
        jax_platform: Optional[str] = None,
        max_workers: int = 4,
    ):
        """
        Initialize the base JAX preprocessor.
        
        Args:
            image_size: Target image size (height, width)
            use_gpu: Whether to use GPU for JAX preprocessing
            jax_platform: Specific JAX platform ('cpu', 'gpu', 'tpu', or None for auto)
            max_workers: Maximum threads for parallel image loading
        """
        self.image_size = image_size
        self.use_gpu = use_gpu
        self.jax_platform = jax_platform
        self.max_workers = max_workers
        
        # Configure JAX device
        self._configure_jax_device()
    
    def _configure_jax_device(self) -> None:
        """
        Configure JAX to use specified device (CPU/GPU/TPU).
        
        Sets up JAX device based on configuration and logs the device being used.
        Gracefully falls back to CPU if requested device is not available.
        """
        try:
            # Get available devices
            devices = jax.devices()
            
            # Determine target platform
            target_platform = None
            if self.jax_platform:
                target_platform = self.jax_platform.lower()
            elif self.use_gpu:
                target_platform = 'gpu'
            
            if target_platform:
                # Try to get devices of the specified type
                try:
                    platform_devices = [d for d in devices if d.platform.lower() == target_platform]
                    if platform_devices:
                        self.device = platform_devices[0]
                        logger.info(f"JAX configured to use {target_platform.upper()}: {self.device}")
                    else:
                        logger.warning(
                            f"No {target_platform.upper()} devices found. "
                            f"Available devices: {[d.platform for d in devices]}. "
                            f"Falling back to default device."
                        )
                        self.device = jax.devices()[0]
                        logger.info(f"Using default device: {self.device}")
                except Exception as e:
                    logger.warning(f"Error selecting {target_platform} device: {e}. Using default device.")
                    self.device = jax.devices()[0]
                    logger.info(f"Using default device: {self.device}")
            else:
                # Use default device (usually CPU)
                self.device = jax.devices()[0]
                logger.info(f"Using default JAX device: {self.device}")
                
        except Exception as e:
            logger.error(f"Error configuring JAX device: {e}. Using default.")
            self.device = jax.devices()[0]
            logger.info(f"Using default device: {self.device}")
    
    def _to_device(self, array: jnp.ndarray) -> jnp.ndarray:
        """
        Transfer array to the configured device.
        
        Args:
            array: JAX array
            
        Returns:
            Array on the configured device
        """
        return jax.device_put(array, self.device)
    
    def load_image_from_path(self, path: str) -> np.ndarray:
        """
        Load image from local file path.
        
        Args:
            path: Local file path
            
        Returns:
            Image as numpy array (H, W, C) in RGB format
        """
        try:
            from PIL import Image
            img = Image.open(path).convert('RGB')
            return np.array(img, dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to load image from {path}: {e}")
            raise
    
    def load_image_from_url(self, url: str, timeout: int = 10) -> np.ndarray:
        """
        Load image from URL.
        
        Args:
            url: Image URL
            timeout: Request timeout in seconds
            
        Returns:
            Image as numpy array (H, W, C) in RGB format
        """
        try:
            import requests
            import io
            from PIL import Image
            
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content)).convert('RGB')
            return np.array(img, dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to load image from {url}: {e}")
            raise
    
    def load_image(self, input_path: str) -> np.ndarray:
        """
        Load image from local path or URL.
        
        Args:
            input_path: Local file path or HTTP(S) URL
            
        Returns:
            Image as numpy array (H, W, C)
        """
        if input_path.startswith(('http://', 'https://')):
            return self.load_image_from_url(input_path)
        else:
            return self.load_image_from_path(input_path)
    
    def load_images_parallel(self, paths: List[str]) -> List[np.ndarray]:
        """
        Load multiple images in parallel using ThreadPoolExecutor.
        
        Args:
            paths: List of image paths or URLs
            
        Returns:
            List of loaded images as numpy arrays
        """
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            images = list(executor.map(self.load_image, paths))
        return images
    
    @abstractmethod
    def _preprocess_single_jax(self, image: jnp.ndarray) -> jnp.ndarray:
        """
        JAX-based preprocessing for a single image.
        
        This method must be implemented by subclasses and should use JAX operations
        for best performance. The method will be JIT-compiled automatically.
        
        Args:
            image: Input image as JAX array (H, W, C)
            
        Returns:
            Preprocessed image as JAX array
        """
        pass
    
    def preprocess_single(self, image: Union[np.ndarray, str]) -> np.ndarray:
        """
        Preprocess a single image.
        
        Args:
            image: Image array or path/URL
            
        Returns:
            Preprocessed image array
        """
        # Load if path
        if isinstance(image, str):
            image = self.load_image(image)
        
        # Convert to JAX array and place on configured device
        jax_image = jnp.array(image, dtype=jnp.float32)
        jax_image = self._to_device(jax_image)
        
        # Preprocess with subclass implementation
        processed = self._preprocess_single_jax(jax_image)
        
        # Convert back to numpy
        return np.array(processed)
    
    def preprocess_batch(
        self,
        images: List[Union[np.ndarray, str]],
    ) -> np.ndarray:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of image arrays or paths/URLs
            
        Returns:
            Batch of preprocessed images
        """
        # Check if we need to load images
        needs_loading = any(isinstance(img, str) for img in images)
        
        if needs_loading:
            # Separate paths and arrays
            paths = [img for img in images if isinstance(img, str)]
            arrays = [img for img in images if not isinstance(img, str)]
            
            # Load images in parallel if there are paths
            if paths:
                loaded_from_paths = self.load_images_parallel(paths)
                # Combine with arrays in original order
                loaded_images = []
                path_idx = 0
                array_idx = 0
                for img in images:
                    if isinstance(img, str):
                        loaded_images.append(loaded_from_paths[path_idx])
                        path_idx += 1
                    else:
                        loaded_images.append(arrays[array_idx])
                        array_idx += 1
            else:
                loaded_images = arrays
        else:
            loaded_images = images
        
        # Stack into batch
        batch = np.stack(loaded_images, axis=0).astype(np.float32)
        
        # Convert to JAX and place on configured device
        jax_batch = jnp.array(batch)
        jax_batch = self._to_device(jax_batch)
        
        # Vectorized preprocessing using vmap
        batch_fn = jax.vmap(self._preprocess_single_jax, in_axes=0)
        processed_batch = batch_fn(jax_batch)
        
        # Convert back to numpy
        return np.array(processed_batch)
    
    def __call__(
        self,
        images: Union[str, np.ndarray, List[Union[str, np.ndarray]]],
    ) -> np.ndarray:
        """
        Preprocess images (single or batch).
        
        Args:
            images: Single image or list of images (paths/URLs or arrays)
            
        Returns:
            Preprocessed image(s) with batch dimension
        """
        if isinstance(images, list):
            return self.preprocess_batch(images)
        else:
            result = self.preprocess_single(images)
            return result[np.newaxis, ...]  # Add batch dimension


# Keep the old name as an alias for backward compatibility
ImagePreprocessor = BaseJAXPreprocessor
