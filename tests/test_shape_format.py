"""
Test script for shape format and dynamic dimensions.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocess_jax import JAXImagePreprocessor
from src.config import ServiceConfig, PreprocessConfig
from loguru import logger


def test_nhwc_format():
    """Test NHWC format (default)."""
    logger.info("Testing NHWC format...")
    
    preprocessor = JAXImagePreprocessor(
        image_size=(224, 224),
        data_format='NHWC',
        cache_compiled=True,
    )
    
    # Create dummy images
    dummy_images = [np.random.rand(256, 256, 3).astype(np.float32) for _ in range(4)]
    
    # Preprocess batch
    processed = preprocessor.preprocess_batch(dummy_images)
    
    expected_shape = (4, 224, 224, 3)  # (B, H, W, C)
    assert processed.shape == expected_shape, f"Expected {expected_shape}, got {processed.shape}"
    logger.info(f"✓ NHWC format test passed: shape = {processed.shape}")


def test_nchw_format():
    """Test NCHW format (for Triton)."""
    logger.info("Testing NCHW format...")
    
    preprocessor = JAXImagePreprocessor(
        image_size=(224, 224),
        data_format='NCHW',
        cache_compiled=True,
    )
    
    # Create dummy images
    dummy_images = [np.random.rand(256, 256, 3).astype(np.float32) for _ in range(4)]
    
    # Preprocess batch
    processed = preprocessor.preprocess_batch(dummy_images)
    
    expected_shape = (4, 3, 224, 224)  # (B, C, H, W)
    assert processed.shape == expected_shape, f"Expected {expected_shape}, got {processed.shape}"
    logger.info(f"✓ NCHW format test passed: shape = {processed.shape}")


def test_custom_image_size():
    """Test custom image sizes."""
    logger.info("Testing custom image sizes...")
    
    test_sizes = [(128, 128), (256, 256), (384, 384)]
    
    for size in test_sizes:
        preprocessor = JAXImagePreprocessor(
            image_size=size,
            data_format='NCHW',
            cache_compiled=True,
        )
        
        # Create dummy image
        dummy_image = np.random.rand(512, 512, 3).astype(np.float32)
        
        # Preprocess single image
        processed = preprocessor.preprocess_single(dummy_image)
        
        expected_shape = (3, size[0], size[1])  # (C, H, W)
        assert processed.shape == expected_shape, f"Expected {expected_shape}, got {processed.shape}"
        logger.info(f"✓ Custom size {size} test passed: shape = {processed.shape}")


def test_single_image_nchw():
    """Test single image with NCHW format."""
    logger.info("Testing single image NCHW format...")
    
    preprocessor = JAXImagePreprocessor(
        image_size=(224, 224),
        data_format='NCHW',
        cache_compiled=True,
    )
    
    # Create dummy image
    dummy_image = np.random.rand(512, 512, 3).astype(np.float32)
    
    # Preprocess single
    processed = preprocessor.preprocess_single(dummy_image)
    
    expected_shape = (3, 224, 224)  # (C, H, W)
    assert processed.shape == expected_shape, f"Expected {expected_shape}, got {processed.shape}"
    logger.info(f"✓ Single image NCHW test passed: shape = {processed.shape}")


def test_config_integration():
    """Test integration with config."""
    logger.info("Testing config integration...")
    
    # Create config with NCHW format
    config = ServiceConfig()
    config.preprocess.data_format = 'NCHW'
    config.preprocess.image_size = (256, 256)
    
    preprocessor = JAXImagePreprocessor(
        image_size=config.preprocess.image_size,
        mean=config.preprocess.mean,
        std=config.preprocess.std,
        data_format=config.preprocess.data_format,
        cache_compiled=True,
    )
    
    # Create dummy images
    dummy_images = [np.random.rand(300, 300, 3).astype(np.float32) for _ in range(2)]
    
    # Preprocess
    processed = preprocessor.preprocess_batch(dummy_images)
    
    expected_shape = (2, 3, 256, 256)  # (B, C, H, W)
    assert processed.shape == expected_shape, f"Expected {expected_shape}, got {processed.shape}"
    logger.info(f"✓ Config integration test passed: shape = {processed.shape}")


def test_warmup_dynamic_sizes():
    """Test that warmup uses dynamic sizes from config."""
    logger.info("Testing warmup with dynamic sizes...")
    
    test_configs = [
        ((128, 128), 'NHWC'),
        ((256, 256), 'NCHW'),
        ((384, 384), 'NCHW'),
    ]
    
    for size, format_type in test_configs:
        preprocessor = JAXImagePreprocessor(
            image_size=size,
            data_format=format_type,
            cache_compiled=True,  # This triggers warmup
        )
        
        # Test preprocessing with the warmed-up functions
        dummy_image = np.random.rand(*size, 3).astype(np.float32)
        processed = preprocessor.preprocess_single(dummy_image)
        
        if format_type == 'NHWC':
            expected_shape = (*size, 3)
        else:  # NCHW
            expected_shape = (3, *size)
        
        assert processed.shape == expected_shape, f"Expected {expected_shape}, got {processed.shape}"
        logger.info(f"✓ Warmup test passed for size {size}, format {format_type}: shape = {processed.shape}")


def main():
    """Run all tests."""
    logger.info("\n" + "="*60)
    logger.info("Starting shape format and dynamic size tests...")
    logger.info("="*60 + "\n")
    
    try:
        test_nhwc_format()
        print()
        
        test_nchw_format()
        print()
        
        test_custom_image_size()
        print()
        
        test_single_image_nchw()
        print()
        
        test_config_integration()
        print()
        
        test_warmup_dynamic_sizes()
        print()
        
        logger.info("="*60)
        logger.info("✓ All tests passed successfully!")
        logger.info("="*60)
        return 0
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
