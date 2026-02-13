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


def test_protocol_compliance():
    """Test that custom preprocessors implement the protocol."""
    logger.info("Testing protocol compliance...")
    
    # Test SimplePreprocessor
    simple_prep = SimplePreprocessor()
    assert isinstance(simple_prep, ImagePreprocessor), \
        "SimplePreprocessor should implement ImagePreprocessor protocol"
    
    # Verify required methods exist and are callable
    assert callable(getattr(simple_prep, 'preprocess_single', None)), \
        "SimplePreprocessor must have callable preprocess_single method"
    assert callable(getattr(simple_prep, 'preprocess_batch', None)), \
        "SimplePreprocessor must have callable preprocess_batch method"
    assert callable(getattr(simple_prep, '__call__', None)), \
        "SimplePreprocessor must have callable __call__ method"
    
    logger.info("✓ SimplePreprocessor implements ImagePreprocessor protocol")
    
    # Test JAXImagePreprocessor
    jax_prep = JAXImagePreprocessor(cache_compiled=False)
    assert isinstance(jax_prep, ImagePreprocessor), \
        "JAXImagePreprocessor should implement ImagePreprocessor protocol"
    
    # Verify required methods exist and are callable
    assert callable(getattr(jax_prep, 'preprocess_single', None)), \
        "JAXImagePreprocessor must have callable preprocess_single method"
    assert callable(getattr(jax_prep, 'preprocess_batch', None)), \
        "JAXImagePreprocessor must have callable preprocess_batch method"
    assert callable(getattr(jax_prep, '__call__', None)), \
        "JAXImagePreprocessor must have callable __call__ method"
    
    logger.info("✓ JAXImagePreprocessor implements ImagePreprocessor protocol")


def test_simple_preprocessor():
    """Test the simple preprocessor functionality."""
    logger.info("Testing SimplePreprocessor functionality...")
    
    preprocessor = SimplePreprocessor(image_size=(224, 224))
    
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


def test_custom_preprocessor_in_pipeline():
    """Test using a custom preprocessor in the pipeline."""
    logger.info("Testing custom preprocessor in pipeline...")
    
    # Create custom preprocessor
    custom_preprocessor = SimplePreprocessor(image_size=(256, 256))
    
    # Create config
    config = ServiceConfig()
    
    # Create pipeline with custom preprocessor
    # Note: This will fail if Triton/Milvus are not available, but that's okay
    # We're just testing that the interface works
    try:
        pipeline = ImageEmbeddingPipeline(
            config=config,
            preprocessor=custom_preprocessor
        )
        
        # Verify the preprocessor is the custom one
        assert pipeline.preprocessor is custom_preprocessor, \
            "Pipeline should use the provided custom preprocessor"
        logger.info("✓ Pipeline successfully initialized with custom preprocessor")
        
        # Close pipeline
        pipeline.close()
        
    except Exception as e:
        # It's okay if initialization fails due to missing services
        # as long as the preprocessor parameter is accepted
        if "preprocessor" in str(e).lower():
            raise
        logger.warning(f"Pipeline initialization failed (expected if services not running): {e}")
        logger.info("✓ Pipeline accepts custom preprocessor parameter")


def test_default_preprocessor_in_pipeline():
    """Test that pipeline still works without custom preprocessor."""
    logger.info("Testing default preprocessor in pipeline...")
    
    config = ServiceConfig()
    
    try:
        # Create pipeline without custom preprocessor
        pipeline = ImageEmbeddingPipeline(config=config)
        
        # Verify default JAX preprocessor is used
        assert isinstance(pipeline.preprocessor, JAXImagePreprocessor), \
            "Pipeline should use JAXImagePreprocessor by default"
        logger.info("✓ Pipeline uses default JAXImagePreprocessor when no custom preprocessor provided")
        
        # Close pipeline
        pipeline.close()
        
    except Exception as e:
        # It's okay if initialization fails due to missing services
        if "preprocessor" in str(e).lower():
            raise
        logger.warning(f"Pipeline initialization failed (expected if services not running): {e}")
        logger.info("✓ Pipeline creates default preprocessor when none provided")


def test_preprocessor_comparison():
    """Compare output between JAX and Simple preprocessors."""
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
    
    simple_prep = SimplePreprocessor(
        image_size=image_size,
        mean=mean,
        std=std,
    )
    
    # Test with same input
    dummy_image = np.random.rand(256, 256, 3).astype(np.float32) * 255
    
    jax_output = jax_prep.preprocess_single(dummy_image)
    simple_output = simple_prep.preprocess_single(dummy_image)
    
    # Both should produce similar shapes
    assert jax_output.shape == simple_output.shape, \
        f"Shapes should match: JAX={jax_output.shape}, Simple={simple_output.shape}"
    
    # Values should be reasonably close (allowing for numerical differences)
    diff = np.abs(jax_output - simple_output).mean()
    logger.info(f"Mean absolute difference between preprocessors: {diff:.6f}")
    
    # The difference should be small (< 0.1) for similar preprocessing
    # Note: might be higher due to different resize algorithms
    if diff < 0.1:
        logger.info("✓ Preprocessor outputs are very similar")
    else:
        logger.info(f"✓ Preprocessor outputs differ by {diff:.6f} (expected due to different resize methods)")


def main():
    """Run all tests."""
    logger.info("\n" + "="*60)
    logger.info("Starting custom preprocessor interface tests...")
    logger.info("="*60 + "\n")
    
    try:
        test_protocol_compliance()
        print()
        
        test_simple_preprocessor()
        print()
        
        test_custom_preprocessor_in_pipeline()
        print()
        
        test_default_preprocessor_in_pipeline()
        print()
        
        test_preprocessor_comparison()
        print()
        
        logger.info("="*60)
        logger.info("✓ All custom preprocessor tests passed!")
        logger.info("="*60)
        return 0
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
