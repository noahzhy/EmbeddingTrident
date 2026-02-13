"""
High-performance JAX-based image preprocessing with jit and vmap.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import List, Tuple, Union, Optional
import numpy as np
from loguru import logger

from .base_preprocessor import BaseJAXPreprocessor


class JAXImagePreprocessor(BaseJAXPreprocessor):
    """
    JAX-accelerated image preprocessing with jit compilation and vectorization.
    
    Features:
    - JIT-compiled resize and normalization
    - Vectorized batch processing with vmap
    - Support for local files and URLs
    - Cached compiled functions for performance
    - Configurable data format (NHWC or NCHW)
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        cache_compiled: bool = True,
        data_format: str = 'NHWC',
        max_workers: int = 4,
        use_gpu: bool = False,
        jax_platform: Optional[str] = None,
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
            max_workers: Maximum number of threads for parallel image loading
            use_gpu: Whether to use GPU for JAX preprocessing (default: False)
            jax_platform: Specific JAX platform to use ('cpu', 'gpu', 'tpu', or None for auto)
        """
        # Initialize base class
        super().__init__(
            image_size=image_size,
            use_gpu=use_gpu,
            jax_platform=jax_platform,
            max_workers=max_workers,
        )
        
        # Additional configuration specific to this preprocessor
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
    
    def _preprocess_single_jax(
        self,
        image: jnp.ndarray,
    ) -> jnp.ndarray:

        image = image.astype(jnp.float32)
        H, W = image.shape[:2]
        S_h, S_w = self.image_size
        C = image.shape[2]
        resized = jax.image.resize(
            image,
            (S_h, S_w, C),
            method="bilinear",
        )

        scale = jnp.minimum(S_h / H, S_w / W)
        valid_h = jnp.maximum((H * scale).astype(jnp.int32), 1)
        valid_w = jnp.maximum((W * scale).astype(jnp.int32), 1)
        top = (S_h - valid_h) // 2
        left = (S_w - valid_w) // 2

        y = jnp.arange(S_h)[:, None]
        x = jnp.arange(S_w)[None, :]

        mask_h = (y >= top) & (y < top + valid_h)
        mask_w = (x >= left) & (x < left + valid_w)
        mask = mask_h & mask_w
        mask = mask[..., None]  # (H,W,1)
        img = resized * mask

        img = img / 255.0
        img = (img - self.mean) / self.std
        return img
    
    def _get_preprocess_single_jitted(self):
        """Get or create JIT-compiled preprocessing function."""
        if not hasattr(self, '_preprocess_jit'):
            self._preprocess_jit = jit(self._preprocess_single_jax)
        return self._preprocess_jit
    
    @staticmethod
    @jit
    def _transpose_single_nchw(image: jnp.ndarray) -> jnp.ndarray:
        """
        JIT-compiled transpose for single image from NHWC to NCHW.
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Transposed image (C, H, W)
        """
        return jnp.transpose(image, (2, 0, 1))
    
    @staticmethod
    @jit
    def _transpose_batch_nchw(batch: jnp.ndarray) -> jnp.ndarray:
        """
        JIT-compiled transpose for batch from NHWC to NCHW.
        
        Args:
            batch: Input batch (B, H, W, C)
            
        Returns:
            Transposed batch (B, C, H, W)
        """
        return jnp.transpose(batch, (0, 3, 1, 2))
    
    def _get_preprocess_batch_vmap(self) -> callable:
        """
        Get or create cached vectorized batch preprocessing function.
        
        Returns:
            Cached vectorized preprocessing function
        """
        if not hasattr(self, '_preprocess_batch_vmap_cached'):
            preprocess_jit = self._get_preprocess_single_jitted()
            self._preprocess_batch_vmap_cached = vmap(preprocess_jit, in_axes=0)
        return self._preprocess_batch_vmap_cached
    
    def _warmup(self) -> None:
        """Warmup JIT compilation with dummy data."""
        # Use configured image_size instead of hardcoded values
        dummy_image = jnp.ones((*self.image_size, 3), dtype=jnp.float32)
        dummy_image = self._to_device(dummy_image)
        preprocess_jit = self._get_preprocess_single_jitted()
        _ = preprocess_jit(dummy_image)
        
        # Warmup batch processing with small batch
        dummy_batch = jnp.ones((4, *self.image_size, 3), dtype=jnp.float32)
        dummy_batch = self._to_device(dummy_batch)
        batch_fn = self._get_preprocess_batch_vmap()
        _ = batch_fn(dummy_batch)
        
        # Warmup transpose functions if using NCHW
        if self.data_format == 'NCHW':
            _ = self._transpose_single_nchw(dummy_image)
            _ = self._transpose_batch_nchw(dummy_batch)
        
        logger.info("JAX functions compiled and cached")
    
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
        
        # Convert to JAX array and place on configured device
        jax_image = jnp.array(image, dtype=jnp.float32)
        jax_image = self._to_device(jax_image)
        
        # Preprocess with jitted function
        preprocess_jit = self._get_preprocess_single_jitted()
        processed = preprocess_jit(jax_image)
        
        # Convert to NCHW if requested using JIT-compiled transpose
        if self.data_format == 'NCHW':
            processed = self._transpose_single_nchw(processed)
        
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
        
        # Stack into batch - pre-allocate for efficiency
        batch = np.stack(loaded_images, axis=0).astype(np.float32)
        
        # Convert to JAX and place on configured device
        jax_batch = jnp.array(batch)
        jax_batch = self._to_device(jax_batch)
        
        # Vectorized preprocessing using cached vmap
        batch_fn = self._get_preprocess_batch_vmap()
        processed_batch = batch_fn(jax_batch)
        
        # Convert to NCHW if requested using JIT-compiled transpose
        if self.data_format == 'NCHW':
            processed_batch = self._transpose_batch_nchw(processed_batch)
        
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
