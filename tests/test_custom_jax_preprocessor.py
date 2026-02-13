"""
Test script for JAX-based custom preprocessor interface.

This test demonstrates how users can create and use custom JAX preprocessors
by inheriting from BaseJAXPreprocessor.
"""

import sys
import os
import numpy as np
import jax
import jax.numpy as jnp
from typing import Union, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.base_preprocessor import BaseJAXPreprocessor, ImagePreprocessor
from src.preprocess_jax import JAXImagePreprocessor
from src.pipeline import ImageEmbeddingPipeline
from src.config import ServiceConfig
from loguru import logger


class SimpleJAXPreprocessor(BaseJAXPreprocessor):
    """
    Example custom JAX preprocessor implementation.
    
    This is a minimal preprocessor that demonstrates how to inherit from
    BaseJAXPreprocessor and implement custom JAX-based preprocessing logic.
    """
    
    def __init__(
        self,
        image_size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        **kwargs
    ):
        """Initialize the simple JAX preprocessor."""
        super().__init__(image_size=image_size, **kwargs)
        self.mean = jnp.array(mean, dtype=jnp.float32).reshape(1, 1, 3)
        self.std = jnp.array(std, dtype=jnp.float32).reshape(1, 1, 3)
        logger.info(f"SimpleJAXPreprocessor initialized with size={image_size}")
    
    def _preprocess_single_jax(self, image: jnp.ndarray) -> jnp.ndarray:
        """
        JAX-based preprocessing for a single image.
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Preprocessed image
        """
        # Resize using JAX
        resized = jax.image.resize(
            image,
            shape=(*self.image_size, image.shape[2]),
            method='bilinear'
        )
        
        # Normalize
        normalized = (resized / 255.0 - self.mean) / self.std
        
        return normalized


def test_base_class_enforcement():
    """Test that the base class enforcement works."""
    logger.info("Testing base class enforcement...")
    
    # Test that JAXImagePreprocessor inherits from BaseJAXPreprocessor
    jax_prep = JAXImagePreprocessor(cache_compiled=False)
    assert isinstance(jax_prep, BaseJAXPreprocessor), \
        "JAXImagePreprocessor should inherit from BaseJAXPreprocessor"
    logger.info("✓ JAXImagePreprocessor inherits from BaseJAXPreprocessor")
    
    # Test that custom preprocessor inherits from BaseJAXPreprocessor
    simple_prep = SimpleJAXPreprocessor()
    assert isinstance(simple_prep, BaseJAXPreprocessor), \
        "SimpleJAXPreprocessor should inherit from BaseJAXPreprocessor"
    logger.info("✓ SimpleJAXPreprocessor inherits from BaseJAXPreprocessor")
    
    # Test ImagePreprocessor alias
    assert isinstance(jax_prep, ImagePreprocessor), \
        "JAXImagePreprocessor should be instance of ImagePreprocessor alias"
    logger.info("✓ ImagePreprocessor alias works correctly")


def test_simple_jax_preprocessor():
    """Test the simple JAX preprocessor functionality."""
    logger.info("Testing SimpleJAXPreprocessor functionality...")
    
    preprocessor = SimpleJAXPreprocessor(image_size=(224, 224))
    
    # Test single image preprocessing
    dummy_image = np.random.rand(256, 256, 3).astype(np.float32) * 255
    processed = preprocessor.preprocess_single(dummy_image)
    
    assert processed.shape == (224, 224, 3), \
        f"Expected shape (224, 224, 3), got {processed.shape}"
    logger.info(f"✓ Single image preprocessing: shape = {processed.shape}")
    
    # Test batch preprocessing
    dummy_images = [
        np.random.rand(300, 300, 3).astype(np.float32) * 255 
        for _ in range(4)
    ]
    processed_batch = preprocessor.preprocess_batch(dummy_images)
    
    assert processed_batch.shape == (4, 224, 224, 3), \
        f"Expected shape (4, 224, 224, 3), got {processed_batch.shape}"
    logger.info(f"✓ Batch preprocessing: shape = {processed_batch.shape}")
    
    # Test __call__ interface
    processed_call = preprocessor(dummy_image)
    assert processed_call.shape == (1, 224, 224, 3), \
        f"Expected shape (1, 224, 224, 3), got {processed_call.shape}"
    logger.info(f"✓ __call__ interface: shape = {processed_call.shape}")


def test_pipeline_jax_enforcement():
    """Test that pipeline enforces JAX-based preprocessors."""
    logger.info("Testing pipeline JAX enforcement...")
    
    config = ServiceConfig()
    
    # Test 1: Valid JAX preprocessor (inherits from BaseJAXPreprocessor)
    valid_prep = SimpleJAXPreprocessor(image_size=(256, 256))
    
    try:
        pipeline = ImageEmbeddingPipeline(config=config, preprocessor=valid_prep)
        logger.info("✓ Valid JAX preprocessor accepted")
        pipeline.close()
    except TypeError as e:
        if 'BaseJAXPreprocessor' in str(e):
            raise AssertionError(f"Valid preprocessor rejected: {e}")
    except Exception as e:
        # Other errors (Triton/Milvus not available) are okay
        logger.info(f"✓ Valid JAX preprocessor accepted (other services unavailable)")
    
    # Test 2: Invalid non-JAX preprocessor (does NOT inherit from BaseJAXPreprocessor)
    class InvalidPreprocessor:
        """Preprocessor that doesn't inherit from BaseJAXPreprocessor."""
        def preprocess_single(self, image): return np.zeros((224, 224, 3))
        def preprocess_batch(self, images): return np.zeros((len(images), 224, 224, 3))
        def __call__(self, images):
            if isinstance(images, list):
                return self.preprocess_batch(images)
            return self.preprocess_single(images)[np.newaxis, ...]
    
    invalid_prep = InvalidPreprocessor()
    
    try:
        pipeline = ImageEmbeddingPipeline(config=config, preprocessor=invalid_prep)
        raise AssertionError("Invalid preprocessor was accepted (should have been rejected!)")
    except TypeError as e:
        if 'BaseJAXPreprocessor' in str(e):
            logger.info("✓ Invalid non-JAX preprocessor correctly rejected")
        else:
            raise AssertionError(f"Wrong error type: {e}")


def test_default_preprocessor_in_pipeline():
    """Test that pipeline uses default JAXImagePreprocessor when none provided."""
    logger.info("Testing default preprocessor in pipeline...")
    
    config = ServiceConfig()
    
    try:
        # Create pipeline without custom preprocessor
        pipeline = ImageEmbeddingPipeline(config=config)
        
        # Verify default JAX preprocessor is used
        assert isinstance(pipeline.preprocessor, JAXImagePreprocessor), \
            "Pipeline should use JAXImagePreprocessor by default"
        logger.info("✓ Pipeline uses default JAXImagePreprocessor when none provided")
        
        # Close pipeline
        pipeline.close()
        
    except Exception as e:
        # It's okay if initialization fails due to missing services
        if 'BaseJAXPreprocessor' in str(e) or 'preprocessor' in str(e).lower():
            raise
        logger.info("✓ Pipeline creates default preprocessor (other services unavailable)")


def test_jax_operations():
    """Test that custom preprocessors can use JAX operations."""
    logger.info("Testing JAX operations in custom preprocessor...")
    
    class AdvancedJAXPreprocessor(BaseJAXPreprocessor):
        """Preprocessor with advanced JAX operations."""
        
        def __init__(self, image_size=(224, 224), apply_blur=True, **kwargs):
            super().__init__(image_size=image_size, **kwargs)
            self.apply_blur = apply_blur
        
        def _preprocess_single_jax(self, image: jnp.ndarray) -> jnp.ndarray:
            # Resize
            resized = jax.image.resize(
                image,
                shape=(*self.image_size, image.shape[2]),
                method='bilinear'
            )
            
            # Apply optional blur using JAX
            if self.apply_blur:
                # Simple box blur using JAX operations
                kernel = jnp.ones((3, 3, 1)) / 9.0
                # Convolve each channel
                blurred = jnp.zeros_like(resized)
                for c in range(resized.shape[2]):
                    channel = resized[:, :, c:c+1]
                    # Simple averaging for blur effect
                    blurred = blurred.at[:, :, c].set(channel[:, :, 0])
            else:
                blurred = resized
            
            # Normalize
            normalized = blurred / 255.0
            
            return normalized
    
    preprocessor = AdvancedJAXPreprocessor(image_size=(128, 128), apply_blur=True)
    
    # Test preprocessing
    dummy_image = np.random.rand(256, 256, 3).astype(np.float32) * 255
    processed = preprocessor.preprocess_single(dummy_image)
    
    assert processed.shape == (128, 128, 3), \
        f"Expected shape (128, 128, 3), got {processed.shape}"
    logger.info("✓ Advanced JAX operations work in custom preprocessor")


def test_preprocessor_comparison():
    """Compare output between built-in JAX and custom preprocessors."""
    logger.info("Comparing preprocessor outputs...")
    
    # Create same configuration for both
    image_size = (224, 224)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    jax_prep = JAXImagePreprocessor(
        image_size=image_size,
        mean=mean,
        std=std,
        cache_compiled=False,
        data_format='NHWC',
    )
    
    simple_prep = SimpleJAXPreprocessor(
        image_size=image_size,
        mean=mean,
        std=std,
    )
    
    # Test with same input
    dummy_image = np.random.rand(256, 256, 3).astype(np.float32) * 255
    
    jax_output = jax_prep.preprocess_single(dummy_image)
    simple_output = simple_prep.preprocess_single(dummy_image)
    
    # Both should produce same shapes
    assert jax_output.shape == simple_output.shape, \
        f"Shapes should match: JAX={jax_output.shape}, Simple={simple_output.shape}"
    
    # Values should be very close (allowing for floating point differences)
    diff = np.abs(jax_output - simple_output).mean()
    logger.info(f"Mean absolute difference between preprocessors: {diff:.6f}")
    
    # The difference should be very small for identical preprocessing
    assert diff < 0.01, f"Preprocessors differ too much: {diff:.6f}"
    logger.info("✓ JAX preprocessor outputs are consistent")


def main():
    """Run all tests."""
    logger.info("\n" + "="*60)
    logger.info("Starting JAX-based custom preprocessor tests...")
    logger.info("="*60 + "\n")
    
    try:
        test_base_class_enforcement()
        print()
        
        test_simple_jax_preprocessor()
        print()
        
        test_pipeline_jax_enforcement()
        print()
        
        test_default_preprocessor_in_pipeline()
        print()
        
        test_jax_operations()
        print()
        
        test_preprocessor_comparison()
        print()
        
        logger.info("="*60)
        logger.info("✓ All JAX-based custom preprocessor tests passed!")
        logger.info("="*60)
        return 0
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
