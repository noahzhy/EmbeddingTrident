"""
High-performance JAX-based image preprocessing with jit and vmap.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import List, Tuple, Union, Optional
import numpy as np
from PIL import Image
import io
import requests
from functools import lru_cache
from loguru import logger


class JAXImagePreprocessor:
    """
    JAX-accelerated image preprocessing with jit compilation and vectorization.
    
    Features:
    - JIT-compiled resize and normalization
    - Vectorized batch processing with vmap
    - Support for local files and URLs
    - Cached compiled functions for performance
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        cache_compiled: bool = True,
        data_format: str = 'NHWC',
    ):
        """
        Initialize the preprocessor.
        
        Args:
            image_size: Target image size (height, width)
            mean: Normalization mean for RGB channels
            std: Normalization std for RGB channels
            cache_compiled: Whether to cache JIT-compiled functions
            data_format: Output data format - 'NHWC' (batch, height, width, channels) 
                        or 'NCHW' (batch, channels, height, width). Default: 'NHWC'
        """
        self.image_size = image_size
        self.mean = jnp.array(mean, dtype=jnp.float32).reshape(1, 1, 3)
        self.std = jnp.array(std, dtype=jnp.float32).reshape(1, 1, 3)
        self.cache_compiled = cache_compiled
        self.data_format = data_format.upper()
        
        if self.data_format not in ['NHWC', 'NCHW']:
            raise ValueError(f"data_format must be 'NHWC' or 'NCHW', got {data_format}")
        
        # Pre-compile functions
        if cache_compiled:
            logger.info("Pre-compiling JAX functions...")
            self._warmup()
    
    def _resize_image_jax(self, image: jnp.ndarray) -> jnp.ndarray:
        """
        JIT-compiled image resize using JAX.
        
        Args:
            image: Input image array (H, W, C)
            
        Returns:
            Resized image array
        """
        return jax.image.resize(
            image,
            shape=(self.image_size[0], self.image_size[1], image.shape[2]),
            method='bilinear',
        )
    
    def _normalize_image_jax(
        self,
        image: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        JIT-compiled image normalization.
        
        Args:
            image: Input image array (H, W, C) in [0, 255]
            
        Returns:
            Normalized image array in [-1, 1] range
        """
        # Convert to [0, 1]
        image = image / 255.0
        # Normalize
        image = (image - self.mean) / self.std
        return image
    
    def _preprocess_single_jax(
        self,
        image: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        JIT-compiled single image preprocessing pipeline.
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Preprocessed image
        """
        # Resize
        resized = jax.image.resize(
            image,
            shape=(self.image_size[0], self.image_size[1], image.shape[2]),
            method='bilinear',
        )
        # Normalize
        normalized = (resized / 255.0 - self.mean) / self.std
        return normalized
    
    def _get_preprocess_single_jitted(self):
        """Get or create JIT-compiled preprocessing function."""
        if not hasattr(self, '_preprocess_jit'):
            self._preprocess_jit = jit(self._preprocess_single_jax)
        return self._preprocess_jit
    
    def _preprocess_batch_vmap(self) -> callable:
        """
        Create vectorized batch preprocessing function using vmap.
        
        Returns:
            Vectorized preprocessing function
        """
        # Get jitted function
        preprocess_jit = self._get_preprocess_single_jitted()
        # Vectorize over the batch dimension
        return vmap(preprocess_jit, in_axes=0)
    
    def _warmup(self) -> None:
        """Warmup JIT compilation with dummy data."""
        # Use configured image_size instead of hardcoded values
        dummy_image = jnp.ones((*self.image_size, 3), dtype=jnp.float32)
        preprocess_jit = self._get_preprocess_single_jitted()
        _ = preprocess_jit(dummy_image)
        
        # Warmup batch processing with small batch
        dummy_batch = jnp.ones((4, *self.image_size, 3), dtype=jnp.float32)
        batch_fn = self._preprocess_batch_vmap()
        _ = batch_fn(dummy_batch)
        
        logger.info("JAX functions compiled and cached")
    
    def load_image_from_path(self, path: str) -> np.ndarray:
        """
        Load image from local file path.
        
        Args:
            path: Local file path
            
        Returns:
            Image as numpy array (H, W, C) in RGB format
        """
        try:
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
    
    def preprocess_single(self, image: Union[np.ndarray, str]) -> np.ndarray:
        """
        Preprocess a single image.
        
        Args:
            image: Image array or path/URL
            
        Returns:
            Preprocessed image (H, W, C) if NHWC format, or (C, H, W) if NCHW format
        """
        if isinstance(image, str):
            image = self.load_image(image)
        
        # Convert to JAX array
        jax_image = jnp.array(image, dtype=jnp.float32)
        
        # Preprocess with jitted function
        preprocess_jit = self._get_preprocess_single_jitted()
        processed = preprocess_jit(jax_image)
        
        # Convert to NCHW if requested
        if self.data_format == 'NCHW':
            processed = jnp.transpose(processed, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        
        # Convert back to numpy
        return np.array(processed)
    
    def preprocess_batch(
        self,
        images: List[Union[np.ndarray, str]],
    ) -> np.ndarray:
        """
        Preprocess a batch of images using vmap for parallelization.
        
        Args:
            images: List of image arrays or paths/URLs
            
        Returns:
            Batch of preprocessed images (B, H, W, C) if NHWC format, 
            or (B, C, H, W) if NCHW format
        """
        # Load all images
        loaded_images = []
        for img in images:
            if isinstance(img, str):
                img = self.load_image(img)
            loaded_images.append(img)
        
        # Stack into batch
        batch = np.stack(loaded_images, axis=0).astype(np.float32)
        
        # Convert to JAX
        jax_batch = jnp.array(batch)
        
        # Vectorized preprocessing
        batch_fn = self._preprocess_batch_vmap()
        processed_batch = batch_fn(jax_batch)
        
        # Convert to NCHW if requested
        if self.data_format == 'NCHW':
            processed_batch = jnp.transpose(processed_batch, (0, 3, 1, 2))  # (B, H, W, C) -> (B, C, H, W)
        
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
            Preprocessed image(s)
        """
        if isinstance(images, list):
            return self.preprocess_batch(images)
        else:
            result = self.preprocess_single(images)
            return result[np.newaxis, ...]  # Add batch dimension
