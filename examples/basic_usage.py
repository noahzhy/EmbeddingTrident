"""
Basic usage example for the image embedding pipeline.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import ImageEmbeddingPipeline
from src.config import ServiceConfig
from loguru import logger


def main():
    """Basic usage example."""
    
    # Configure logging
    logger.info("Starting basic usage example...")
    
    # Option 1: Use default configuration
    # pipeline = ImageEmbeddingPipeline()
    
    # Option 2: Load from YAML file
    # config = ServiceConfig.from_yaml('../configs/config.yaml')
    # pipeline = ImageEmbeddingPipeline(config)
    
    # Option 3: Load from environment variables
    config = ServiceConfig.from_env()
    
    # Create pipeline (use context manager for automatic cleanup)
    with ImageEmbeddingPipeline(config) as pipeline:
        
        # Check health
        logger.info("Checking service health...")
        health = pipeline.health_check()
        logger.info(f"Service health: {health}")
        
        # Create a collection
        collection_name = "demo_images"
        logger.info(f"Creating collection '{collection_name}'...")
        pipeline.create_collection(
            name=collection_name,
            dim=512,
            description="Demo image embeddings",
        )
        
        # Example image paths (replace with your actual images)
        image_paths = [
            "/path/to/image1.jpg",
            "/path/to/image2.jpg",
            "/path/to/image3.jpg",
        ]
        
        # Or use URLs
        image_urls = [
            "https://example.com/image1.jpg",
            "https://example.com/image2.jpg",
        ]
        
        # Example 1: Extract embeddings only
        logger.info("Extracting embeddings...")
        embeddings = pipeline.embed_images(image_paths[:1])
        logger.info(f"Extracted embedding shape: {embeddings.shape}")
        
        # Example 2: Insert images into Milvus
        logger.info("Inserting images into Milvus...")
        ids = [f"img_{i}" for i in range(len(image_paths))]
        metadata = [
            {"source": "local", "category": "nature"},
            {"source": "local", "category": "urban"},
            {"source": "local", "category": "portrait"},
        ]
        
        inserted_ids = pipeline.insert_images(
            inputs=image_paths,
            ids=ids,
            metadata=metadata,
            collection_name=collection_name,
        )
        logger.info(f"Inserted {len(inserted_ids)} images")
        
        # Example 3: Search for similar images
        logger.info("Searching for similar images...")
        query_image = image_paths[0]
        results = pipeline.search_images(
            query_input=query_image,
            topk=5,
            collection_name=collection_name,
        )
        
        logger.info(f"Search results:")
        for i, result in enumerate(results, 1):
            logger.info(
                f"  {i}. ID: {result['id']}, "
                f"Score: {result['score']:.4f}, "
                f"Metadata: {result.get('metadata', {})}"
            )
        
        # Example 4: Search with filter
        logger.info("Searching with filter...")
        results = pipeline.search_images(
            query_input=query_image,
            topk=5,
            filter_expr="metadata['category'] == 'nature'",
            collection_name=collection_name,
        )
        logger.info(f"Found {len(results)} filtered results")
        
        # Example 5: Get collection stats
        stats = pipeline.get_collection_stats(collection_name)
        logger.info(f"Collection stats: {stats}")
        
        # Example 6: Delete embeddings
        logger.info("Deleting embeddings...")
        pipeline.delete_by_ids([ids[0]], collection_name=collection_name)
        
        # Example 7: List all collections
        collections = pipeline.list_collections()
        logger.info(f"All collections: {collections}")
        
        # Cleanup: Delete collection
        logger.info(f"Cleaning up: deleting collection '{collection_name}'...")
        pipeline.delete_collection(collection_name)
        
    logger.info("Example completed!")


if __name__ == "__main__":
    main()
