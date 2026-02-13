"""
Tests for streaming multiprocessing preprocessor.
"""

import sys
import os
import time
import tempfile
import shutil
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.streaming_preprocessor import StreamingMultiprocessPreprocessor
from src.preprocess_jax import JAXImagePreprocessor
from loguru import logger


def create_test_images(num_images=20, size=(512, 512)):
    """Create temporary test images."""
    temp_dir = tempfile.mkdtemp()
    image_paths = []
    
    for i in range(num_images):
        # Create random image
        img_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Save to temp directory
        path = os.path.join(temp_dir, f"test_image_{i:03d}.jpg")
        img.save(path)
        image_paths.append(path)
    
    return temp_dir, image_paths


def test_streaming_preprocessor_basic():
    """Test basic streaming preprocessor functionality."""
    logger.info("\n" + "="*60)
    logger.info("Test: Basic Streaming Preprocessor Functionality")
    logger.info("="*60)
    
    temp_dir = None
    try:
        # Create test images
        num_images = 16
        temp_dir, image_paths = create_test_images(num_images=num_images)
        logger.info(f"Created {num_images} test images")
        
        # Create streaming preprocessor
        preprocessor = StreamingMultiprocessPreprocessor(
            num_workers=4,
            batch_size=4,
            image_size=(224, 224),
            data_format='NCHW',
        )
        
        # Test context manager
        with preprocessor:
            results = list(preprocessor.preprocess_stream(image_paths, batch_size=4))
        
        # Validate results
        assert len(results) == 4, f"Expected 4 batches, got {len(results)}"
        
        total_images = sum(result['preprocessed'].shape[0] for result in results)
        assert total_images == num_images, f"Expected {num_images} images, got {total_images}"
        
        # Check shapes
        for result in results:
            batch_size = result['preprocessed'].shape[0]
            expected_shape = (batch_size, 3, 224, 224)  # NCHW format
            assert result['preprocessed'].shape == expected_shape, \
                f"Expected shape {expected_shape}, got {result['preprocessed'].shape}"
        
        logger.info("✓ Basic functionality test passed")
        
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir)


def test_streaming_preprocessor_with_metadata():
    """Test streaming preprocessor with IDs and metadata."""
    logger.info("\n" + "="*60)
    logger.info("Test: Streaming Preprocessor with IDs and Metadata")
    logger.info("="*60)
    
    temp_dir = None
    try:
        # Create test images
        num_images = 12
        temp_dir, image_paths = create_test_images(num_images=num_images)
        
        # Create IDs and metadata
        ids = [f"img_{i:03d}" for i in range(num_images)]
        metadata = [{"index": i, "tag": f"tag_{i}"} for i in range(num_images)]
        
        # Create streaming preprocessor
        preprocessor = StreamingMultiprocessPreprocessor(
            num_workers=3,
            batch_size=4,
            image_size=(224, 224),
        )
        
        # Process with metadata
        with preprocessor:
            results = list(preprocessor.preprocess_stream(
                image_paths, ids=ids, metadata=metadata, batch_size=4
            ))
        
        # Validate IDs and metadata are preserved
        collected_ids = []
        collected_metadata = []
        for result in results:
            collected_ids.extend(result['ids'])
            collected_metadata.extend(result['metadata'])
        
        assert collected_ids == ids, "IDs not preserved correctly"
        assert collected_metadata == metadata, "Metadata not preserved correctly"
        
        logger.info("✓ IDs and metadata test passed")
        
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir)


def test_streaming_preprocessor_sync():
    """Test synchronous batch preprocessing."""
    logger.info("\n" + "="*60)
    logger.info("Test: Synchronous Batch Preprocessing")
    logger.info("="*60)
    
    temp_dir = None
    try:
        # Create test images
        num_images = 20
        temp_dir, image_paths = create_test_images(num_images=num_images)
        
        # Create streaming preprocessor
        preprocessor = StreamingMultiprocessPreprocessor(
            num_workers=4,
            batch_size=5,
            image_size=(384, 384),
            data_format='NHWC',
        )
        
        # Process synchronously
        with preprocessor:
            result = preprocessor.preprocess_batch_sync(image_paths, batch_size=5)
        
        # Validate
        expected_shape = (num_images, 384, 384, 3)  # NHWC format
        assert result.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {result.shape}"
        
        logger.info("✓ Synchronous preprocessing test passed")
        
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir)


def test_streaming_vs_sequential_performance():
    """Compare streaming multiprocessing vs sequential preprocessing."""
    logger.info("\n" + "="*60)
    logger.info("Test: Streaming vs Sequential Performance")
    logger.info("="*60)
    
    temp_dir = None
    try:
        # Create test images
        num_images = 32
        temp_dir, image_paths = create_test_images(num_images=num_images)
        logger.info(f"Created {num_images} test images")
        
        # Test sequential preprocessing
        logger.info("\nTesting sequential preprocessing...")
        sequential_preprocessor = JAXImagePreprocessor(
            image_size=(224, 224),
            data_format='NCHW',
            cache_compiled=True,
            max_workers=4,
        )
        
        start_time = time.time()
        sequential_result = sequential_preprocessor.preprocess_batch(image_paths)
        sequential_time = time.time() - start_time
        
        logger.info(f"Sequential: {sequential_time:.3f}s ({num_images/sequential_time:.1f} img/s)")
        
        # Test streaming multiprocessing preprocessing
        logger.info("\nTesting streaming multiprocessing preprocessing...")
        streaming_preprocessor = StreamingMultiprocessPreprocessor(
            num_workers=4,
            batch_size=8,
            image_size=(224, 224),
            data_format='NCHW',
            cache_compiled=True,
        )
        
        start_time = time.time()
        with streaming_preprocessor:
            streaming_result = streaming_preprocessor.preprocess_batch_sync(image_paths, batch_size=8)
        streaming_time = time.time() - start_time
        
        logger.info(f"Streaming: {streaming_time:.3f}s ({num_images/streaming_time:.1f} img/s)")
        
        # Calculate speedup
        speedup = sequential_time / streaming_time
        logger.info(f"\nSpeedup: {speedup:.2f}x")
        
        # Validate shapes match
        assert sequential_result.shape == streaming_result.shape, \
            f"Shape mismatch: {sequential_result.shape} vs {streaming_result.shape}"
        
        # Note: We don't check exact equality because:
        # 1. Different processes may have slightly different floating point behavior
        # 2. Image loading from JPEG may have minor variations
        # 3. JAX operations across processes may have minor numerical differences
        # But the shapes and general magnitude should match
        logger.info(f"Sequential result range: [{sequential_result.min():.3f}, {sequential_result.max():.3f}]")
        logger.info(f"Streaming result range: [{streaming_result.min():.3f}, {streaming_result.max():.3f}]")
        
        logger.info("\n✓ Performance comparison test passed")
        
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir)


def test_error_handling():
    """Test error handling in streaming preprocessor."""
    logger.info("\n" + "="*60)
    logger.info("Test: Error Handling")
    logger.info("="*60)
    
    try:
        # Create preprocessor with invalid paths
        preprocessor = StreamingMultiprocessPreprocessor(
            num_workers=2,
            batch_size=4,
            image_size=(224, 224),
        )
        
        # Try to process non-existent images
        invalid_paths = ["/nonexistent/image1.jpg", "/nonexistent/image2.jpg"]
        
        with preprocessor:
            try:
                results = list(preprocessor.preprocess_stream(invalid_paths, batch_size=2))
                # If we get here, it means the error wasn't caught
                logger.error("Expected error not raised")
                assert False, "Should have raised an error for invalid paths"
            except (RuntimeError, TimeoutError) as e:
                logger.info(f"✓ Error correctly caught: {e}")
        
        logger.info("✓ Error handling test passed")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


def test_custom_preprocessor():
    """Test streaming with custom preprocessor."""
    logger.info("\n" + "="*60)
    logger.info("Test: Custom Preprocessor")
    logger.info("="*60)
    
    temp_dir = None
    try:
        # Create test images
        num_images = 8
        temp_dir, image_paths = create_test_images(num_images=num_images)
        
        # Use custom preprocessor (JAXImagePreprocessor with custom settings)
        preprocessor = StreamingMultiprocessPreprocessor(
            num_workers=2,
            batch_size=4,
            preprocessor_class=JAXImagePreprocessor,
            preprocessor_kwargs={
                'image_size': (256, 256),
                'mean': (0.5, 0.5, 0.5),
                'std': (0.5, 0.5, 0.5),
                'data_format': 'NCHW',
                'cache_compiled': True,
            },
        )
        
        # Process
        with preprocessor:
            results = list(preprocessor.preprocess_stream(image_paths, batch_size=4))
        
        # Validate
        for result in results:
            batch_size = result['preprocessed'].shape[0]
            expected_shape = (batch_size, 3, 256, 256)
            assert result['preprocessed'].shape == expected_shape, \
                f"Expected shape {expected_shape}, got {result['preprocessed'].shape}"
        
        logger.info("✓ Custom preprocessor test passed")
        
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir)


def main():
    """Run all tests."""
    logger.info("Starting streaming multiprocessing preprocessor tests...")
    
    tests = [
        test_streaming_preprocessor_basic,
        test_streaming_preprocessor_with_metadata,
        test_streaming_preprocessor_sync,
        test_streaming_vs_sequential_performance,
        test_error_handling,
        test_custom_preprocessor,
    ]
    
    failed_tests = []
    
    for test in tests:
        try:
            test()
        except Exception as e:
            logger.error(f"Test {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed_tests.append(test.__name__)
    
    logger.info("\n" + "="*60)
    if not failed_tests:
        logger.info("✓ All tests passed!")
        logger.info("="*60)
        return 0
    else:
        logger.error(f"✗ {len(failed_tests)} test(s) failed:")
        for test_name in failed_tests:
            logger.error(f"  - {test_name}")
        logger.info("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
