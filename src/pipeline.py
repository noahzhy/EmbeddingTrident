"""
End-to-end pipeline orchestration for image embedding service.
"""

import numpy as np
from typing import List, Dict, Optional, Union, Any, Tuple
from loguru import logger
import time
import gevent
from gevent import queue
from concurrent.futures import ThreadPoolExecutor

from .base_preprocessor import BaseJAXPreprocessor
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
    
    def __init__(
        self, 
        config: Optional[ServiceConfig] = None,
        preprocessor: Optional[BaseJAXPreprocessor] = None,
    ):
        """
        Initialize the pipeline.
        
        Args:
            config: Service configuration
            preprocessor: Optional custom JAX preprocessor inheriting from BaseJAXPreprocessor.
                         If None, uses default JAXImagePreprocessor with config settings.
                         Custom preprocessors must inherit from BaseJAXPreprocessor to ensure
                         JAX-based high-performance preprocessing.
        """
        if config is None:
            config = ServiceConfig()
        
        self.config = config
        
        # Initialize components
        logger.info("Initializing image embedding pipeline...")
        
        # Preprocessor - use custom if provided, otherwise create JAX preprocessor
        if preprocessor is not None:
            # Validate that the preprocessor inherits from BaseJAXPreprocessor
            if not isinstance(preprocessor, BaseJAXPreprocessor):
                raise TypeError(
                    f"Custom preprocessor must inherit from BaseJAXPreprocessor, "
                    f"got {type(preprocessor).__name__}. "
                    f"This ensures JAX-based high-performance preprocessing."
                )
            logger.info(f"Using custom preprocessor: {type(preprocessor).__name__}")
            self.preprocessor = preprocessor
        else:
            logger.info("Using default JAXImagePreprocessor")
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
            vector_field=config.milvus.vector_field,
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
    
    def insert_images_async(
        self,
        inputs: List[str],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        collection_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        preprocess_workers: Optional[int] = None,
        embedding_workers: Optional[int] = None,
        insert_batch_size: Optional[int] = None,
        queue_maxsize: Optional[int] = None,
    ) -> List[str]:
        """
        Extract embeddings and insert into Milvus using async pipeline.
        
        Architecture:
            Main thread: JAX preprocessing → Producer greenlet → Embedding workers → Queue → Async Milvus inserter
        
        This prevents GPU from waiting on database operations while keeping JAX operations in the main thread.
        Note: Uses gevent greenlets instead of OS threads for better cooperative multitasking.
        
        Args:
            inputs: List of image paths or URLs
            ids: List of unique IDs for images
            metadata: Optional metadata for each image
            collection_name: Target collection name
            batch_size: Batch size for preprocessing and embedding
            preprocess_workers: Number of preprocessing worker greenlets (default: from config)
            embedding_workers: Number of embedding worker greenlets (default: from config)
            insert_batch_size: Batch size for Milvus insertion (default: from config)
            queue_maxsize: Maximum size of the embedding queue (default: from config)
            
        Returns:
            List of inserted IDs
        """
        if len(inputs) != len(ids):
            raise ValueError("Number of inputs must match number of IDs")
        
        # Use config defaults if not provided
        if batch_size is None:
            batch_size = self.config.preprocess.batch_size
        if preprocess_workers is None:
            preprocess_workers = self.config.async_pipeline.preprocess_workers
        if embedding_workers is None:
            embedding_workers = self.config.async_pipeline.embedding_workers
        if insert_batch_size is None:
            insert_batch_size = self.config.async_pipeline.insert_batch_size
        if queue_maxsize is None:
            queue_maxsize = self.config.async_pipeline.queue_maxsize
        
        start_time = time.time()
        
        # Preprocess all images in the main thread (JAX operations must run in the same thread)
        logger.info(f"Preprocessing {len(inputs)} images in batches of {batch_size}")
        preprocessed_batches = []
        try:
            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                batch_metadata = metadata[i:i + batch_size] if metadata else None
                
                # Preprocess batch in main thread
                preprocessed = self.preprocessor.preprocess_batch(batch_inputs)
                
                preprocessed_batches.append({
                    'preprocessed': preprocessed,
                    'ids': batch_ids,
                    'metadata': batch_metadata,
                    'batch_idx': i // batch_size,
                })
                
                logger.debug(f"Preprocessed batch {i // batch_size}, size={len(batch_inputs)}")
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            raise RuntimeError(f"Preprocessing failed: {e}")
        
        logger.info(f"Preprocessing complete, {len(preprocessed_batches)} batches ready")
        
        # Queues for async pipeline
        preprocess_queue = queue.Queue(maxsize=queue_maxsize)
        embedding_queue = queue.Queue(maxsize=queue_maxsize)
        
        # Sentinel value to signal completion
        SENTINEL = object()
        
        # Shared state
        error_container = []
        
        # Producer: Feed preprocessed batches to embedding workers
        def producer():
            try:
                logger.info(f"Producer started: feeding {len(preprocessed_batches)} preprocessed batches")
                for batch_data in preprocessed_batches:
                    # Put preprocessed batch in queue for embedding workers
                    preprocess_queue.put(batch_data)
                    logger.debug(f"Producer: queued batch {batch_data['batch_idx']}")
                
                logger.info("Producer: all batches queued")
            except Exception as e:
                logger.error(f"Producer error: {e}")
                error_container.append(e)
            finally:
                # Send sentinel to each embedding worker
                for _ in range(embedding_workers):
                    preprocess_queue.put(SENTINEL)
        
        # Embedding workers: Generate embeddings from preprocessed data
        def embedding_worker(worker_id):
            try:
                logger.info(f"Embedding worker {worker_id} started")
                while True:
                    item = preprocess_queue.get()
                    
                    # Check for sentinel
                    if item is SENTINEL:
                        preprocess_queue.task_done()
                        logger.info(f"Embedding worker {worker_id}: received sentinel, stopping")
                        break
                    
                    # Generate embeddings
                    embeddings = self.triton_client.infer(
                        item['preprocessed'],
                        normalize=True,
                    )
                    
                    # Put in queue for inserter
                    embedding_queue.put({
                        'embeddings': embeddings,
                        'ids': item['ids'],
                        'metadata': item['metadata'],
                        'batch_idx': item['batch_idx'],
                    })
                    
                    logger.debug(f"Embedding worker {worker_id}: processed batch {item['batch_idx']}, embeddings shape={embeddings.shape}")
                    preprocess_queue.task_done()
                
                logger.info(f"Embedding worker {worker_id}: processing complete")
            except Exception as e:
                logger.error(f"Embedding worker {worker_id} error: {e}")
                error_container.append(e)
            finally:
                # Signal completion by putting sentinel in embedding queue
                embedding_queue.put(SENTINEL)
        
        # Milvus inserter: Batch insert embeddings asynchronously
        def milvus_inserter():
            try:
                logger.info(f"Milvus inserter started with batch size {insert_batch_size}")
                batch_embeddings = []
                batch_ids_list = []
                batch_metadata_list = []
                inserted_count = 0
                sentinels_received = 0
                
                while sentinels_received < embedding_workers:
                    item = embedding_queue.get()
                    
                    # Check for sentinel
                    if item is SENTINEL:
                        sentinels_received += 1
                        embedding_queue.task_done()
                        logger.debug(f"Milvus inserter: received sentinel {sentinels_received}/{embedding_workers}")
                        continue
                    
                    # Accumulate items
                    batch_embeddings.append(item['embeddings'])
                    batch_ids_list.extend(item['ids'])
                    if item['metadata']:
                        batch_metadata_list.extend(item['metadata'])
                    
                    logger.debug(f"Milvus inserter: accumulated batch {item['batch_idx']}, total buffered={len(batch_ids_list)}")
                    
                    # Insert when batch is full
                    if len(batch_ids_list) >= insert_batch_size:
                        combined_embeddings = np.concatenate(batch_embeddings, axis=0)
                        combined_metadata = batch_metadata_list if batch_metadata_list else None
                        
                        self.milvus_client.insert_embeddings(
                            ids=batch_ids_list,
                            embeddings=combined_embeddings,
                            metadata=combined_metadata,
                            collection_name=collection_name,
                        )
                        
                        inserted_count += len(batch_ids_list)
                        logger.info(f"Milvus inserter: inserted batch of {len(batch_ids_list)} embeddings, total={inserted_count}")
                        
                        # Reset batch
                        batch_embeddings = []
                        batch_ids_list = []
                        batch_metadata_list = []
                    
                    embedding_queue.task_done()
                
                # Insert remaining items
                if batch_ids_list:
                    combined_embeddings = np.concatenate(batch_embeddings, axis=0)
                    combined_metadata = batch_metadata_list if batch_metadata_list else None
                    
                    self.milvus_client.insert_embeddings(
                        ids=batch_ids_list,
                        embeddings=combined_embeddings,
                        metadata=combined_metadata,
                        collection_name=collection_name,
                    )
                    
                    inserted_count += len(batch_ids_list)
                    logger.info(f"Milvus inserter: inserted final batch of {len(batch_ids_list)} embeddings, total={inserted_count}")
                
                logger.info(f"Milvus inserter complete: total inserted {inserted_count}")
            except Exception as e:
                logger.error(f"Milvus inserter error: {e}")
                error_container.append(e)
        
        # Start all greenlets (gevent coroutines)
        producer_greenlet = gevent.spawn(producer)
        embedding_greenlets = [
            gevent.spawn(embedding_worker, i)
            for i in range(embedding_workers)
        ]
        inserter_greenlet = gevent.spawn(milvus_inserter)
        
        # Wait for all greenlets to complete
        gevent.joinall([producer_greenlet] + embedding_greenlets + [inserter_greenlet])
        logger.debug("All greenlets completed")
        
        # Check for errors
        if error_container:
            raise RuntimeError(f"Async pipeline failed: {error_container[0]}")
        
        total_time = time.time() - start_time
        throughput = len(inputs) / total_time
        
        logger.info(
            f"Async pipeline: Inserted {len(ids)} images in {total_time:.3f}s "
            f"({throughput:.1f} images/sec)"
        )
        
        return ids
    
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
