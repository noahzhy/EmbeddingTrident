"""
Async batch processing example demonstrating the optimized pipeline.

This example shows how the async pipeline prevents GPU from waiting on database operations
by implementing a producer-consumer architecture with:
- Producer threads for preprocessing
- Embedding worker pool for GPU operations
- Queue for decoupling embedding and insertion
- Async Milvus inserter for background batch insertion
"""

import sys
import os
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import ImageEmbeddingPipeline
from src.config import ServiceConfig
from loguru import logger


def generate_dummy_images(num_images: int, save_dir: str) -> list:
    """
    Generate dummy images for testing.
    
    Args:
        num_images: Number of images to generate
        save_dir: Directory to save images
        
    Returns:
        List of image paths
    """
    from PIL import Image
    
    os.makedirs(save_dir, exist_ok=True)
    
    image_paths = []
    for i in range(num_images):
        # Create random image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Save image
        path = os.path.join(save_dir, f"test_image_{i:04d}.jpg")
        img.save(path)
        image_paths.append(path)
    
    return image_paths


def compare_sync_vs_async(
    pipeline: ImageEmbeddingPipeline,
    num_images: int = 1000,
    batch_size: int = 32,
):
    """
    Compare sync vs async pipeline performance.
    
    Args:
        pipeline: Pipeline instance
        num_images: Number of images to process
        batch_size: Batch size for processing
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Comparing sync vs async pipeline with {num_images} images")
    logger.info(f"{'='*80}\n")
    
    # Generate test images
    logger.info("Generating test images...")
    image_paths = generate_dummy_images(num_images, "/tmp/test_images_async")
    ids = [f"img_{i:06d}" for i in range(num_images)]
    metadata = [{"batch": i // batch_size} for i in range(num_images)]
    
    # Test 1: Synchronous pipeline (baseline)
    collection_name_sync = "sync_benchmark"
    try:
        pipeline.delete_collection(collection_name_sync)
    except:
        pass
    
    pipeline.create_collection(collection_name_sync, dim=pipeline.config.milvus.embedding_dim)
    
    logger.info("\n--- Test 1: Synchronous Pipeline ---")
    start_time = time.time()
    pipeline.insert_images(
        inputs=image_paths,
        ids=ids,
        metadata=metadata,
        collection_name=collection_name_sync,
        batch_size=batch_size,
    )
    sync_time = time.time() - start_time
    sync_throughput = num_images / sync_time
    
    logger.info(f"Sync Pipeline Performance:")
    logger.info(f"  Total time: {sync_time:.3f}s")
    logger.info(f"  Throughput: {sync_throughput:.1f} images/sec")
    
    # Test 2: Asynchronous pipeline (optimized)
    collection_name_async = "async_benchmark"
    try:
        pipeline.delete_collection(collection_name_async)
    except:
        pass
    
    pipeline.create_collection(collection_name_async, dim=pipeline.config.milvus.embedding_dim)
    
    logger.info("\n--- Test 2: Asynchronous Pipeline ---")
    start_time = time.time()
    pipeline.insert_images_async(
        inputs=image_paths,
        ids=ids,
        metadata=metadata,
        collection_name=collection_name_async,
        batch_size=batch_size,
    )
    async_time = time.time() - start_time
    async_throughput = num_images / async_time
    
    logger.info(f"Async Pipeline Performance:")
    logger.info(f"  Total time: {async_time:.3f}s")
    logger.info(f"  Throughput: {async_throughput:.1f} images/sec")
    
    # Compare
    speedup = sync_time / async_time
    logger.info(f"\n{'='*80}")
    logger.info(f"RESULTS SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Synchronous:  {sync_time:.3f}s ({sync_throughput:.1f} images/sec)")
    logger.info(f"Asynchronous: {async_time:.3f}s ({async_throughput:.1f} images/sec)")
    logger.info(f"Speedup:      {speedup:.2f}x")
    logger.info(f"{'='*80}\n")
    
    # Verify data integrity
    logger.info("Verifying data integrity...")
    sync_stats = pipeline.get_collection_stats(collection_name_sync)
    async_stats = pipeline.get_collection_stats(collection_name_async)
    
    logger.info(f"Sync collection: {sync_stats['num_entities']} entities")
    logger.info(f"Async collection: {async_stats['num_entities']} entities")
    
    if sync_stats['num_entities'] == async_stats['num_entities'] == num_images:
        logger.info("✓ Data integrity verified: All images inserted correctly")
    else:
        logger.error("✗ Data integrity issue: Entity counts don't match")
    
    # Cleanup
    logger.info("\nCleaning up...")
    pipeline.delete_collection(collection_name_sync)
    pipeline.delete_collection(collection_name_async)
    
    # Remove test images
    import shutil
    shutil.rmtree("/tmp/test_images_async")
    
    logger.info("Benchmark complete!")


def main():
    """Run async batch processing benchmark."""
    
    logger.info("Starting async batch processing benchmark...")
    
    # Load configuration
    config = ServiceConfig.from_env()
    
    # Optional: Customize async pipeline settings
    config.async_pipeline.preprocess_workers = 2
    config.async_pipeline.embedding_workers = 1  # Usually 1 for GPU
    config.async_pipeline.insert_batch_size = 100
    config.async_pipeline.queue_maxsize = 100
    
    # Create pipeline
    with ImageEmbeddingPipeline(config) as pipeline:
        
        # Check health
        health = pipeline.health_check()
        if not all(health.values()):
            logger.error(f"Service not fully healthy: {health}")
            logger.error("Make sure Triton and Milvus are running")
            return
        
        # Run comparison
        try:
            compare_sync_vs_async(
                pipeline,
                num_images=1000,
                batch_size=32,
            )
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("All benchmarks completed!")


if __name__ == "__main__":
    main()
