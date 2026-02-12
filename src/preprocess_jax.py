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
    ):
        """
        Initialize the preprocessor.
        
        Args:
            image_size: Target image size (height, width)
            mean: Normalization mean for RGB channels
            std: Normalization std for RGB channels
            cache_compiled: Whether to cache JIT-compiled functions
        """
        self.image_size = image_size
        self.mean = jnp.array(mean, dtype=jnp.float32).reshape(1, 1, 3)
        self.std = jnp.array(std, dtype=jnp.float32).reshape(1, 1, 3)
        self.cache_compiled = cache_compiled
        
        # Pre-compile functions
        if cache_compiled:
            logger.info("Pre-compiling JAX functions...")
            self._warmup()
    
    @staticmethod
    @jit
    def _resize_image_jax(image: jnp.ndarray, target_size: Tuple[int, int]) -> jnp.ndarray:
        """
        JIT-compiled image resize using JAX.
        
        Args:
            image: Input image array (H, W, C)
            target_size: Target (height, width)
            
        Returns:
            Resized image array
        """
        return jax.image.resize(
            image,
            shape=(target_size[0], target_size[1], image.shape[2]),
            method='bilinear',
        )
    
    @staticmethod
    @jit
    def _normalize_image_jax(
        image: jnp.ndarray,
        mean: jnp.ndarray,
        std: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        JIT-compiled image normalization.
        
        Args:
            image: Input image array (H, W, C) in [0, 255]
            mean: Normalization mean
            std: Normalization std
            
        Returns:
            Normalized image array in [-1, 1] range
        """
        # Convert to [0, 1]
        image = image / 255.0
        # Normalize
        image = (image - mean) / std
        return image
    
    @staticmethod
    @jit
    def _preprocess_single_jax(
        image: jnp.ndarray,
        target_size: Tuple[int, int],
        mean: jnp.ndarray,
        std: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        JIT-compiled single image preprocessing pipeline.
        
        Args:
            image: Input image (H, W, C)
            target_size: Target size
            mean: Normalization mean
            std: Normalization std
            
        Returns:
            Preprocessed image
        """
        # Resize
        resized = jax.image.resize(
            image,
            shape=(target_size[0], target_size[1], image.shape[2]),
            method='bilinear',
        )
        # Normalize
        normalized = (resized / 255.0 - mean) / std
        return normalized
    
    def _preprocess_batch_vmap(self) -> callable:
        """
        Create vectorized batch preprocessing function using vmap.
        
        Returns:
            Vectorized preprocessing function
        """
        # Vectorize over the batch dimension
        return vmap(
            lambda img: self._preprocess_single_jax(
                img,
                self.image_size,
                self.mean,
                self.std,
            ),
            in_axes=0,
        )
    
    def _warmup(self) -> None:
        """Warmup JIT compilation with dummy data."""
        dummy_image = jnp.ones((224, 224, 3), dtype=jnp.float32)
        _ = self._preprocess_single_jax(
            dummy_image,
            self.image_size,
            self.mean,
            self.std,
        )
        
        # Warmup batch processing
        dummy_batch = jnp.ones((4, 224, 224, 3), dtype=jnp.float32)
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
            Preprocessed image (H, W, C)
        """
        if isinstance(image, str):
            image = self.load_image(image)
        
        # Convert to JAX array
        jax_image = jnp.array(image, dtype=jnp.float32)
        
        # Preprocess
        processed = self._preprocess_single_jax(
            jax_image,
            self.image_size,
            self.mean,
            self.std,
        )
        
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
            Batch of preprocessed images (B, H, W, C)
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
