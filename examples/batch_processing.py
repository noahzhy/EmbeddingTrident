"""
Batch processing example with performance monitoring.
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


def benchmark_pipeline(
    pipeline: ImageEmbeddingPipeline,
    num_images: int = 100,
    batch_size: int = 32,
):
    """
    Benchmark the pipeline performance.
    
    Args:
        pipeline: Pipeline instance
        num_images: Number of images to process
        batch_size: Batch size for processing
    """
    logger.info(f"Benchmarking with {num_images} images, batch_size={batch_size}")
    
    # Generate test images
    logger.info("Generating test images...")
    image_paths = generate_dummy_images(num_images, "/tmp/test_images")
    
    # Benchmark embedding extraction
    logger.info("Benchmarking embedding extraction...")
    start_time = time.time()
    embeddings = pipeline.embed_images(image_paths, batch_size=batch_size)
    embedding_time = time.time() - start_time
    
    throughput = num_images / embedding_time
    avg_latency = embedding_time / num_images * 1000
    
    logger.info(f"Embedding Extraction Performance:")
    logger.info(f"  Total time: {embedding_time:.3f}s")
    logger.info(f"  Throughput: {throughput:.1f} images/sec")
    logger.info(f"  Avg latency: {avg_latency:.2f}ms per image")
    logger.info(f"  Embedding shape: {embeddings.shape}")
    
    # Benchmark insert
    collection_name = "benchmark_collection"
    pipeline.create_collection(collection_name, dim=embeddings.shape[1])
    
    logger.info("Benchmarking insert operation...")
    ids = [f"bench_{i:06d}" for i in range(num_images)]
    metadata = [{"batch": i // batch_size} for i in range(num_images)]
    
    start_time = time.time()
    pipeline.insert_images(
        inputs=image_paths,
        ids=ids,
        metadata=metadata,
        collection_name=collection_name,
        batch_size=batch_size,
    )
    insert_time = time.time() - start_time
    
    insert_throughput = num_images / insert_time
    
    logger.info(f"Insert Performance:")
    logger.info(f"  Total time: {insert_time:.3f}s")
    logger.info(f"  Throughput: {insert_throughput:.1f} vectors/sec")
    
    # Benchmark search
    logger.info("Benchmarking search operation...")
    num_queries = 100
    query_times = []
    
    for i in range(num_queries):
        query_embedding = embeddings[i % len(embeddings)]
        
        start_time = time.time()
        results = pipeline.search_images(
            query_input=query_embedding,
            topk=10,
            collection_name=collection_name,
        )
        query_time = (time.time() - start_time) * 1000
        query_times.append(query_time)
    
    avg_query_time = np.mean(query_times)
    p50_query_time = np.percentile(query_times, 50)
    p95_query_time = np.percentile(query_times, 95)
    p99_query_time = np.percentile(query_times, 99)
    
    logger.info(f"Search Performance ({num_queries} queries):")
    logger.info(f"  Avg latency: {avg_query_time:.2f}ms")
    logger.info(f"  P50 latency: {p50_query_time:.2f}ms")
    logger.info(f"  P95 latency: {p95_query_time:.2f}ms")
    logger.info(f"  P99 latency: {p99_query_time:.2f}ms")
    
    # Cleanup
    logger.info("Cleaning up...")
    pipeline.delete_collection(collection_name)
    
    # Remove test images
    import shutil
    shutil.rmtree("/tmp/test_images")
    
    logger.info("Benchmark complete!")


def main():
    """Run batch processing benchmark."""
    
    logger.info("Starting batch processing benchmark...")
    
    # Load configuration
    config = ServiceConfig.from_env()
    
    # Create pipeline
    with ImageEmbeddingPipeline(config) as pipeline:
        
        # Check health
        health = pipeline.health_check()
        if not all(health.values()):
            logger.error(f"Service not fully healthy: {health}")
            logger.error("Make sure Triton and Milvus are running")
            return
        
        # Run benchmarks with different batch sizes
        for batch_size in [8, 16, 32]:
            logger.info(f"\n{'='*60}")
            logger.info(f"Batch size: {batch_size}")
            logger.info(f"{'='*60}\n")
            
            try:
                benchmark_pipeline(
                    pipeline,
                    num_images=1000,
                    batch_size=batch_size,
                )
            except Exception as e:
                logger.error(f"Benchmark failed for batch_size={batch_size}: {e}")
    
    logger.info("All benchmarks completed!")


if __name__ == "__main__":
    main()
