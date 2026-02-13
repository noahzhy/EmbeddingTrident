"""
End-to-end pipeline orchestration for image embedding service.
"""

import numpy as np
from typing import List, Dict, Optional, Union, Any, Tuple
from loguru import logger
import time

from .preprocess_jax import JAXImagePreprocessor
from .triton_client import TritonClient
from .milvus_client import MilvusClient
from .config import ServiceConfig


class ImageEmbeddingPipeline:
    """
    High-performance end-to-end pipeline for image embedding.
    
    Pipeline flow:
    1. Load images (local files or URLs)
    2. JAX preprocessing (jit + vmap)
    3. Triton inference (embedding extraction)
    4. L2 normalization
    5. Insert into Milvus
    6. Support vector search
    
    Optimizations:
    - Batch processing throughout
    - JIT-compiled JAX operations
    - Vectorized preprocessing with vmap
    - Minimized data transfer overhead
    """
    
    def __init__(self, config: Optional[ServiceConfig] = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Service configuration
        """
        if config is None:
            config = ServiceConfig()
        
        self.config = config
        
        # Initialize components
        logger.info("Initializing image embedding pipeline...")
        
        # JAX preprocessor
        self.preprocessor = JAXImagePreprocessor(
            image_size=config.preprocess.image_size,
            mean=config.preprocess.mean,
            std=config.preprocess.std,
            cache_compiled=config.cache_compiled_functions,
            data_format=config.preprocess.data_format,
            max_workers=config.preprocess.num_workers,
            use_gpu=config.preprocess.use_gpu,
            jax_platform=config.preprocess.jax_platform,
        )
        
        # Triton client
        self.triton_client = TritonClient(
            url=config.triton.url,
            model_name=config.triton.model_name,
            model_version=config.triton.model_version,
            protocol=config.triton.protocol,
            timeout=config.triton.timeout,
            max_retries=config.triton.max_retries,
            retry_delay=config.triton.retry_delay,
            input_name=config.triton.input_name,
            output_name=config.triton.output_name,
        )
        
        # Milvus client
        self.milvus_client = MilvusClient(
            host=config.milvus.host,
            port=config.milvus.port,
            collection_name=config.milvus.collection_name,
            embedding_dim=config.milvus.embedding_dim,
            index_type=config.milvus.index_type,
            metric_type=config.milvus.metric_type,
            nlist=config.milvus.nlist,
            nprobe=config.milvus.nprobe,
            M=config.milvus.M,
            efConstruction=config.milvus.efConstruction,
        )
        
        logger.info("Pipeline initialization complete")
    
    def embed_images(
        self,
        inputs: List[str],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Extract embeddings from images.
        
        Args:
            inputs: List of image paths or URLs
            batch_size: Batch size for processing (uses config default if None)
            
        Returns:
            Embedding vectors (N, D)
        """
        if batch_size is None:
            batch_size = self.config.preprocess.batch_size
        
        start_time = time.time()
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i + batch_size]
            
            # Stage 1: JAX preprocessing
            preprocess_start = time.time()
            preprocessed = self.preprocessor.preprocess_batch(batch_inputs)
            preprocess_time = time.time() - preprocess_start
            
            logger.debug(
                f"Preprocessed batch of {len(batch_inputs)} images "
                f"in {preprocess_time:.3f}s"
            )
            
            # Stage 2: Triton inference
            inference_start = time.time()
            embeddings = self.triton_client.infer(
                preprocessed,
                normalize=True,
            )
            inference_time = time.time() - inference_start
            
            logger.debug(
                f"Generated embeddings for batch of {len(batch_inputs)} images "
                f"in {inference_time:.3f}s"
            )
            
            all_embeddings.append(embeddings)
        
        # Concatenate all embeddings
        final_embeddings = np.concatenate(all_embeddings, axis=0)
        
        total_time = time.time() - start_time
        throughput = len(inputs) / total_time
        
        logger.info(
            f"Generated {len(inputs)} embeddings in {total_time:.3f}s "
            f"({throughput:.1f} images/sec)"
        )
        
        return final_embeddings
    
    def insert_images(
        self,
        inputs: List[str],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        collection_name: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """
        Extract embeddings and insert into Milvus.
        
        Args:
            inputs: List of image paths or URLs
            ids: List of unique IDs for images
            metadata: Optional metadata for each image
            collection_name: Target collection name
            batch_size: Batch size for processing
            
        Returns:
            List of inserted IDs
        """
        if len(inputs) != len(ids):
            raise ValueError("Number of inputs must match number of IDs")
        
        start_time = time.time()
        
        # Extract embeddings
        logger.info(f"Extracting embeddings for {len(inputs)} images...")
        embeddings = self.embed_images(inputs, batch_size=batch_size)
        
        # Insert into Milvus
        logger.info(f"Inserting {len(embeddings)} embeddings into Milvus...")
        inserted_ids = self.milvus_client.insert_embeddings(
            ids=ids,
            embeddings=embeddings,
            metadata=metadata,
            collection_name=collection_name,
        )
        
        total_time = time.time() - start_time
        throughput = len(inputs) / total_time
        
        logger.info(
            f"Inserted {len(inserted_ids)} images in {total_time:.3f}s "
            f"({throughput:.1f} images/sec)"
        )
        
        return inserted_ids
    
    def search_images(
        self,
        query_input: Union[str, np.ndarray],
        topk: int = 10,
        filter_expr: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar images.
        
        Args:
            query_input: Query image path/URL or embedding vector
            topk: Number of results to return
            filter_expr: Optional filter expression
            collection_name: Target collection name
            
        Returns:
            List of search results
        """
        start_time = time.time()
        
        # Get query embedding
        if isinstance(query_input, str):
            logger.debug(f"Generating query embedding from image: {query_input}")
            query_embedding = self.embed_images([query_input])[0]
        else:
            query_embedding = query_input
        
        # Search in Milvus
        results = self.milvus_client.search_topk(
            query_embedding=query_embedding,
            topk=topk,
            filter_expr=filter_expr,
            collection_name=collection_name,
        )
        
        search_time = time.time() - start_time
        
        logger.info(
            f"Search completed in {search_time * 1000:.2f}ms, "
            f"found {len(results)} results"
        )
        
        return results
    
    def create_collection(
        self,
        name: str,
        dim: Optional[int] = None,
        description: str = "",
    ) -> None:
        """
        Create a new collection.
        
        Args:
            name: Collection name
            dim: Embedding dimension (uses config default if None)
            description: Collection description
        """
        if dim is None:
            dim = self.config.milvus.embedding_dim
        
        self.milvus_client.create_collection(
            name=name,
            dim=dim,
            description=description,
        )
    
    def delete_collection(self, name: str) -> None:
        """
        Delete a collection.
        
        Args:
            name: Collection name
        """
        self.milvus_client.drop_collection(name)
    
    def list_collections(self) -> List[str]:
        """
        List all collections.
        
        Returns:
            List of collection names
        """
        return self.milvus_client.list_collections()
    
    def delete_by_ids(
        self,
        ids: List[str],
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Delete embeddings by IDs.
        
        Args:
            ids: List of IDs to delete
            collection_name: Target collection name
        """
        self.milvus_client.delete_by_ids(ids, collection_name)
    
    def delete_by_filter(
        self,
        expr: str,
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Delete embeddings by filter expression.
        
        Args:
            expr: Filter expression
            collection_name: Target collection name
        """
        self.milvus_client.delete_by_filter(expr, collection_name)
    
    def get_collection_stats(
        self,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Args:
            collection_name: Collection name
            
        Returns:
            Collection statistics
        """
        return self.milvus_client.get_collection_stats(collection_name)
    
    def health_check(self) -> Dict[str, bool]:
        """
        Check health of all components.
        
        Returns:
            Dictionary with component health status
        """
        health = {
            "preprocessor": True,  # Always available
            "triton": False,
            "milvus": False,
        }
        
        # Check Triton
        try:
            if self.triton_client.client.is_server_live():
                health["triton"] = True
        except Exception as e:
            logger.warning(f"Triton health check failed: {e}")
        
        # Check Milvus
        try:
            _ = self.milvus_client.list_collections()
            health["milvus"] = True
        except Exception as e:
            logger.warning(f"Milvus health check failed: {e}")
        
        return health
    
    def close(self) -> None:
        """Close all client connections."""
        logger.info("Closing pipeline connections...")
        
        try:
            self.triton_client.close()
        except Exception as e:
            logger.warning(f"Error closing Triton client: {e}")
        
        try:
            self.milvus_client.disconnect()
        except Exception as e:
            logger.warning(f"Error closing Milvus client: {e}")
        
        logger.info("Pipeline closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
