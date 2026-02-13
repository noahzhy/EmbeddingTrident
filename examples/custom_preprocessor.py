"""
Example: Using a custom JAX preprocessor with the ImageEmbeddingPipeline.

This example demonstrates how to create and use your own custom preprocessor
that inherits from BaseJAXPreprocessor for high-performance JAX-based preprocessing.
"""

import sys
import os
import numpy as np
import jax
import jax.numpy as jnp
from typing import Union, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import BaseJAXPreprocessor, ImageEmbeddingPipeline, ServiceConfig
from loguru import logger


class CustomJAXPreprocessor(BaseJAXPreprocessor):
    """
    Example custom JAX preprocessor with additional preprocessing steps.
    
    This preprocessor demonstrates how to:
    - Inherit from BaseJAXPreprocessor
    - Implement custom JAX-based preprocessing logic
    - Add custom parameters and augmentation
    - Leverage JAX's JIT compilation and vmap for performance
    """
    
    def __init__(
        self,
        image_size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        grayscale=False,
        contrast_adjustment=1.0,
        **kwargs
    ):
        """
        Initialize custom JAX preprocessor.
        
        Args:
            image_size: Target size for images
            mean: Normalization mean
            std: Normalization std
            grayscale: Convert images to grayscale
            contrast_adjustment: Contrast adjustment factor (1.0 = no change)
            **kwargs: Additional arguments passed to BaseJAXPreprocessor
        """
        super().__init__(image_size=image_size, **kwargs)
        
        self.mean = jnp.array(mean, dtype=jnp.float32).reshape(1, 1, 3)
        self.std = jnp.array(std, dtype=jnp.float32).reshape(1, 1, 3)
        self.grayscale = grayscale
        self.contrast_adjustment = contrast_adjustment
        
        logger.info(
            f"CustomJAXPreprocessor initialized: "
            f"size={image_size}, grayscale={grayscale}, "
            f"contrast={contrast_adjustment}"
        )
    
    def _preprocess_single_jax(self, image: jnp.ndarray) -> jnp.ndarray:
        """
        JAX-based preprocessing for a single image.
        
        This method uses JAX operations for best performance.
        It will be automatically JIT-compiled and can be vectorized with vmap.
        
        Args:
            image: Input image (H, W, C) in [0, 255] range
            
        Returns:
            Preprocessed image
        """
        # 1. Resize using JAX
        resized = jax.image.resize(
            image,
            shape=(*self.image_size, image.shape[2]),
            method='bilinear'
        )
        
        # 2. Optional grayscale conversion using JAX
        if self.grayscale:
            # Use standard RGB to grayscale weights
            gray = 0.299 * resized[:, :, 0] + 0.587 * resized[:, :, 1] + 0.114 * resized[:, :, 2]
            # Replicate to 3 channels
            resized = jnp.stack([gray, gray, gray], axis=-1)
        
        # 3. Normalize to [0, 1]
        normalized = resized / 255.0
        
        # 4. Apply contrast adjustment using JAX
        if self.contrast_adjustment != 1.0:
            # Adjust contrast around mean
            mean_val = jnp.mean(normalized)
            normalized = mean_val + self.contrast_adjustment * (normalized - mean_val)
            normalized = jnp.clip(normalized, 0.0, 1.0)
        
        # 5. Standard normalization
        normalized = (normalized - self.mean) / self.std
        
        return normalized


def main():
    """
    Example: Using custom JAX preprocessor with the pipeline.
    """
    logger.info("Custom JAX Preprocessor Example")
    logger.info("="*60)
    
    # Step 1: Create your custom JAX preprocessor
    logger.info("\n1. Creating custom JAX preprocessor...")
    custom_preprocessor = CustomJAXPreprocessor(
        image_size=(256, 256),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        grayscale=False,  # Set to True for grayscale preprocessing
        contrast_adjustment=1.2,  # Increase contrast by 20%
        use_gpu=False,  # Set to True for GPU acceleration
    )
    
    # Step 2: Load configuration
    logger.info("\n2. Loading configuration...")
    config = ServiceConfig.from_env()
    
    # Step 3: Create pipeline with custom preprocessor
    logger.info("\n3. Creating pipeline with custom JAX preprocessor...")
    try:
        pipeline = ImageEmbeddingPipeline(
            config=config,
            preprocessor=custom_preprocessor,  # Pass your custom JAX preprocessor
        )
        
        logger.info("✓ Pipeline created successfully with custom JAX preprocessor!")
        
        # Step 4: Use the pipeline normally
        logger.info("\n4. Using the pipeline...")
        
        # Example: Extract embeddings
        # Note: You'll need actual image paths or URLs
        image_paths = [
            "/path/to/your/image1.jpg",
            "/path/to/your/image2.jpg",
        ]
        
        logger.info(f"Processing {len(image_paths)} images...")
        # embeddings = pipeline.embed_images(image_paths)
        # logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        
        # The rest of the pipeline works the same way:
        # - Insert images: pipeline.insert_images(...)
        # - Search: pipeline.search_images(...)
        # - etc.
        
        logger.info("\n✓ Example completed!")
        
        # Clean up
        pipeline.close()
        
    except TypeError as e:
        if 'BaseJAXPreprocessor' in str(e):
            logger.error(
                f"Preprocessor rejected: {e}\n"
                "Custom preprocessors must inherit from BaseJAXPreprocessor!"
            )
        else:
            raise
    except Exception as e:
        logger.error(f"Pipeline initialization failed: {e}")
        logger.info(
            "\nNote: This is expected if Triton/Milvus services are not running.\n"
            "The example shows how to pass a custom JAX preprocessor to the pipeline."
        )
    
    # Alternative: Test preprocessor independently
    logger.info("\n" + "="*60)
    logger.info("Testing custom JAX preprocessor independently...")
    logger.info("="*60)
    
    # Create dummy image
    dummy_image = np.random.rand(512, 512, 3).astype(np.float32) * 255
    logger.info(f"Input image shape: {dummy_image.shape}")
    
    # Preprocess single image
    processed = custom_preprocessor.preprocess_single(dummy_image)
    logger.info(f"Preprocessed single image shape: {processed.shape}")
    
    # Preprocess batch (leverages JAX vmap for parallelization)
    dummy_batch = [dummy_image.copy() for _ in range(4)]
    processed_batch = custom_preprocessor.preprocess_batch(dummy_batch)
    logger.info(f"Preprocessed batch shape: {processed_batch.shape}")
    
    logger.info("\n✓ Custom JAX preprocessor working correctly!")
    logger.info(
        "\nKey benefits of JAX-based preprocessing:"
        "\n- JIT compilation for fast execution"
        "\n- Automatic vectorization with vmap"
        "\n- GPU acceleration support"
        "\n- Type-safe and high-performance"
    )


if __name__ == "__main__":
    main()
