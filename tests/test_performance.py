"""
Performance benchmark tests for JIT and vmap optimizations.
"""

import sys
import os
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocess_jax import JAXImagePreprocessor
from loguru import logger


def benchmark_preprocessing():
    """Benchmark preprocessing performance."""
    logger.info("\n" + "="*60)
    logger.info("Benchmarking JAX Preprocessing Performance")
    logger.info("="*60)
    
    # Test configurations
    image_sizes = [(224, 224), (384, 384)]
    batch_sizes = [8, 16, 32]
    num_iterations = 3
    
    for image_size in image_sizes:
        logger.info(f"\n--- Image Size: {image_size} ---")
        
        for batch_size in batch_sizes:
            logger.info(f"\nBatch Size: {batch_size}")
            
            # Create preprocessor with JIT compilation
            preprocessor = JAXImagePreprocessor(
                image_size=image_size,
                data_format='NCHW',
                cache_compiled=True,
                max_workers=4,
            )
            
            # Generate dummy images
            dummy_images = [
                np.random.rand(512, 512, 3).astype(np.float32) 
                for _ in range(batch_size)
            ]
            
            # Warmup run
            _ = preprocessor.preprocess_batch(dummy_images)
            
            # Benchmark runs
            times = []
            for i in range(num_iterations):
                start = time.time()
                result = preprocessor.preprocess_batch(dummy_images)
                elapsed = time.time() - start
                times.append(elapsed)
                
                # Validate output shape
                expected_shape = (batch_size, 3, *image_size)
                assert result.shape == expected_shape, \
                    f"Expected {expected_shape}, got {result.shape}"
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            throughput = batch_size / avg_time
            
            logger.info(f"  Average time: {avg_time*1000:.2f} ms ± {std_time*1000:.2f} ms")
            logger.info(f"  Throughput: {throughput:.1f} images/sec")
            logger.info(f"  Per-image latency: {avg_time*1000/batch_size:.2f} ms")


def benchmark_transpose():
    """Benchmark JIT-compiled transpose vs regular transpose."""
    logger.info("\n" + "="*60)
    logger.info("Benchmarking JIT Transpose Performance")
    logger.info("="*60)
    
    batch_size = 32
    image_size = (224, 224)
    num_iterations = 100
    
    preprocessor = JAXImagePreprocessor(
        image_size=image_size,
        data_format='NCHW',
        cache_compiled=True,
    )
    
    # Create test data
    import jax.numpy as jnp
    test_batch = jnp.ones((batch_size, *image_size, 3), dtype=jnp.float32)
    
    # Warmup
    _ = preprocessor._transpose_batch_nchw(test_batch)
    
    # Benchmark JIT transpose
    start = time.time()
    for _ in range(num_iterations):
        result_jit = preprocessor._transpose_batch_nchw(test_batch)
        result_jit.block_until_ready()  # Ensure computation completes
    jit_time = (time.time() - start) / num_iterations
    
    # Benchmark regular transpose
    start = time.time()
    for _ in range(num_iterations):
        result_regular = jnp.transpose(test_batch, (0, 3, 1, 2))
        result_regular.block_until_ready()  # Ensure computation completes
    regular_time = (time.time() - start) / num_iterations
    
    logger.info(f"JIT transpose: {jit_time*1000:.3f} ms")
    logger.info(f"Regular transpose: {regular_time*1000:.3f} ms")
    logger.info(f"Speedup: {regular_time/jit_time:.2f}x")
    
    # Validate correctness
    assert np.allclose(result_jit, result_regular), "Transpose results don't match!"
    logger.info("✓ Correctness validated")


def benchmark_parallel_loading():
    """Benchmark parallel vs sequential image loading."""
    logger.info("\n" + "="*60)
    logger.info("Benchmarking Parallel Image Loading")
    logger.info("="*60)
    
    # Create dummy image files
    import tempfile
    from PIL import Image
    
    num_images = 16
    temp_dir = tempfile.mkdtemp()
    image_paths = []
    
    try:
        # Create test images
        for i in range(num_images):
            img = Image.fromarray(
                np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            )
            path = os.path.join(temp_dir, f"test_image_{i}.jpg")
            img.save(path)
            image_paths.append(path)
        
        # Test with parallel loading
        preprocessor_parallel = JAXImagePreprocessor(
            image_size=(224, 224),
            max_workers=4,
        )
        
        start = time.time()
        loaded_parallel = preprocessor_parallel.load_images_parallel(image_paths)
        parallel_time = time.time() - start
        
        logger.info(f"Parallel loading ({num_images} images): {parallel_time*1000:.2f} ms")
        logger.info(f"Throughput: {num_images/parallel_time:.1f} images/sec")
        
        # Test with sequential loading
        start = time.time()
        loaded_sequential = [preprocessor_parallel.load_image(p) for p in image_paths]
        sequential_time = time.time() - start
        
        logger.info(f"Sequential loading ({num_images} images): {sequential_time*1000:.2f} ms")
        logger.info(f"Speedup: {sequential_time/parallel_time:.2f}x")
        
        # Validate
        assert len(loaded_parallel) == len(loaded_sequential) == num_images
        logger.info("✓ All images loaded successfully")
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)


def benchmark_vmap_caching():
    """Benchmark cached vmap vs recreating vmap each time."""
    logger.info("\n" + "="*60)
    logger.info("Benchmarking vmap Caching")
    logger.info("="*60)
    
    batch_size = 32
    num_iterations = 50
    
    preprocessor = JAXImagePreprocessor(
        image_size=(224, 224),
        cache_compiled=True,
    )
    
    dummy_batch = np.random.rand(batch_size, 256, 256, 3).astype(np.float32)
    import jax.numpy as jnp
    jax_batch = jnp.array(dummy_batch)
    
    # Benchmark with cached vmap
    start = time.time()
    for _ in range(num_iterations):
        batch_fn = preprocessor._get_preprocess_batch_vmap()
        result = batch_fn(jax_batch)
        result.block_until_ready()  # Ensure computation completes
    cached_time = (time.time() - start) / num_iterations
    
    logger.info(f"Cached vmap: {cached_time*1000:.3f} ms per iteration")
    logger.info(f"Throughput: {batch_size/cached_time:.1f} images/sec")


def main():
    """Run all benchmarks."""
    logger.info("Starting performance benchmarks...")
    
    try:
        benchmark_preprocessing()
        benchmark_transpose()
        benchmark_parallel_loading()
        benchmark_vmap_caching()
        
        logger.info("\n" + "="*60)
        logger.info("✓ All benchmarks completed successfully!")
        logger.info("="*60)
        return 0
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
