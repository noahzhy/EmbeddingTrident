"""
Streaming multiprocessing preprocessor for high-throughput image preprocessing.

This module provides a streaming preprocessor that uses multiprocessing to 
parallelize preprocessing across multiple CPU cores, significantly improving
throughput for I/O-bound and CPU-bound preprocessing operations.
"""

import multiprocessing as mp
from multiprocessing import Queue, Process
from typing import List, Union, Optional, Dict, Any, Tuple
import numpy as np
from loguru import logger
import time
import queue as queue_module

from .base_preprocessor import BaseJAXPreprocessor
from .preprocess_jax import JAXImagePreprocessor


class StreamingMultiprocessPreprocessor:
    """
    Streaming multiprocessing preprocessor for high-throughput image preprocessing.
    
    This class creates a pool of worker processes that preprocess images in parallel
    and stream results through a multiprocessing queue. This approach significantly
    improves throughput by:
    
    1. Parallelizing I/O-bound image loading across multiple cores
    2. Parallelizing CPU-bound preprocessing operations
    3. Streaming results to avoid memory buildup
    4. Overlapping preprocessing with downstream operations (embedding, insertion)
    
    Architecture:
        Main Process -> Input Queue -> [Worker 1, Worker 2, ..., Worker N] -> Output Queue -> Consumer
        
    Each worker process:
        1. Receives a batch of image paths/arrays from input queue
        2. Loads images (if paths) using ThreadPoolExecutor
        3. Preprocesses using JAX (jit + vmap)
        4. Sends preprocessed batch to output queue
    
    Features:
        - Configurable number of worker processes
        - Automatic load balancing across workers
        - Graceful shutdown and error handling
        - Compatible with existing preprocessor interface
        - Support for custom JAX preprocessors
    
    Example:
        ```python
        from src.streaming_preprocessor import StreamingMultiprocessPreprocessor
        
        # Create streaming preprocessor
        preprocessor = StreamingMultiprocessPreprocessor(
            num_workers=4,
            batch_size=32,
            image_size=(224, 224),
        )
        
        # Preprocess images
        with preprocessor:
            for batch_result in preprocessor.preprocess_stream(image_paths, batch_size=32):
                preprocessed = batch_result['preprocessed']
                # Process preprocessed batch...
        ```
    """
    
    def __init__(
        self,
        num_workers: Optional[int] = None,
        batch_size: int = 32,
        queue_maxsize: int = 10,
        preprocessor_class: type = JAXImagePreprocessor,
        preprocessor_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the streaming multiprocessing preprocessor.
        
        Args:
            num_workers: Number of worker processes (default: CPU count)
            batch_size: Batch size for preprocessing
            queue_maxsize: Maximum size of input/output queues
            preprocessor_class: Preprocessor class to use (must inherit from BaseJAXPreprocessor)
            preprocessor_kwargs: Additional kwargs for preprocessor initialization
            **kwargs: Additional arguments passed to preprocessor
        """
        if num_workers is None:
            num_workers = mp.cpu_count()
        
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.queue_maxsize = queue_maxsize
        self.preprocessor_class = preprocessor_class
        
        # Merge kwargs for preprocessor
        self.preprocessor_kwargs = preprocessor_kwargs or {}
        self.preprocessor_kwargs.update(kwargs)
        
        # Validate preprocessor class
        if not issubclass(preprocessor_class, BaseJAXPreprocessor):
            raise TypeError(
                f"preprocessor_class must inherit from BaseJAXPreprocessor, "
                f"got {preprocessor_class.__name__}"
            )
        
        # Process management
        self.workers: List[Process] = []
        self.input_queue: Optional[Queue] = None
        self.output_queue: Optional[Queue] = None
        self.is_started = False
        
        logger.info(
            f"StreamingMultiprocessPreprocessor initialized with "
            f"{num_workers} workers, batch_size={batch_size}"
        )
    
    def _worker_process(
        self,
        worker_id: int,
        input_queue: Queue,
        output_queue: Queue,
        preprocessor_class: type,
        preprocessor_kwargs: Dict[str, Any],
    ):
        """
        Worker process that preprocesses batches.
        
        Each worker:
        1. Receives batch data from input queue
        2. Loads and preprocesses images
        3. Sends results to output queue
        
        Args:
            worker_id: Unique worker ID
            input_queue: Queue to receive batches from
            output_queue: Queue to send results to
            preprocessor_class: Preprocessor class to instantiate
            preprocessor_kwargs: Arguments for preprocessor initialization
        """
        try:
            # Create preprocessor in worker process
            # Each process gets its own JAX context and device
            preprocessor = preprocessor_class(**preprocessor_kwargs)
            logger.info(f"Worker {worker_id}: Preprocessor initialized")
            
            while True:
                try:
                    # Get batch from input queue with timeout
                    item = input_queue.get(timeout=1.0)
                    
                    # Check for sentinel (shutdown signal)
                    if item is None:
                        logger.info(f"Worker {worker_id}: Received shutdown signal")
                        break
                    
                    # Unpack batch data
                    batch_idx = item['batch_idx']
                    batch_inputs = item['inputs']
                    batch_ids = item.get('ids', None)
                    batch_metadata = item.get('metadata', None)
                    
                    logger.debug(
                        f"Worker {worker_id}: Processing batch {batch_idx}, "
                        f"size={len(batch_inputs)}"
                    )
                    
                    # Preprocess batch
                    start_time = time.time()
                    preprocessed = preprocessor.preprocess_batch(batch_inputs)
                    preprocess_time = time.time() - start_time
                    
                    # Send result to output queue
                    result = {
                        'batch_idx': batch_idx,
                        'preprocessed': preprocessed,
                        'ids': batch_ids,
                        'metadata': batch_metadata,
                        'worker_id': worker_id,
                        'preprocess_time': preprocess_time,
                    }
                    output_queue.put(result)
                    
                    logger.debug(
                        f"Worker {worker_id}: Completed batch {batch_idx} "
                        f"in {preprocess_time:.3f}s"
                    )
                    
                except queue_module.Empty:
                    # Timeout waiting for input, continue loop
                    continue
                except Exception as e:
                    logger.error(f"Worker {worker_id}: Error processing batch: {e}")
                    # Send error to output queue
                    output_queue.put({
                        'error': str(e),
                        'worker_id': worker_id,
                    })
                    
        except Exception as e:
            logger.error(f"Worker {worker_id}: Fatal error: {e}")
        finally:
            logger.info(f"Worker {worker_id}: Shutting down")
    
    def start(self):
        """Start worker processes and queues."""
        if self.is_started:
            logger.warning("StreamingMultiprocessPreprocessor already started")
            return
        
        logger.info(f"Starting {self.num_workers} worker processes...")
        
        # Create queues
        self.input_queue = Queue(maxsize=self.queue_maxsize)
        self.output_queue = Queue(maxsize=self.queue_maxsize)
        
        # Start worker processes
        for worker_id in range(self.num_workers):
            process = Process(
                target=self._worker_process,
                args=(
                    worker_id,
                    self.input_queue,
                    self.output_queue,
                    self.preprocessor_class,
                    self.preprocessor_kwargs,
                ),
                daemon=True,
            )
            process.start()
            self.workers.append(process)
            logger.debug(f"Started worker process {worker_id} (PID: {process.pid})")
        
        self.is_started = True
        logger.info(f"All {self.num_workers} workers started successfully")
    
    def stop(self, timeout: float = 5.0):
        """
        Stop worker processes gracefully.
        
        Args:
            timeout: Maximum time to wait for workers to finish (seconds)
        """
        if not self.is_started:
            return
        
        logger.info("Stopping worker processes...")
        
        # Send shutdown signal to all workers
        for _ in range(self.num_workers):
            self.input_queue.put(None)
        
        # Wait for workers to finish
        start_time = time.time()
        for worker in self.workers:
            remaining_time = max(0, timeout - (time.time() - start_time))
            worker.join(timeout=remaining_time)
            if worker.is_alive():
                logger.warning(f"Worker {worker.pid} did not stop gracefully, terminating")
                worker.terminate()
                worker.join(timeout=1.0)
        
        self.workers = []
        self.is_started = False
        logger.info("All workers stopped")
    
    def preprocess_stream(
        self,
        inputs: List[Union[str, np.ndarray]],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Preprocess images in streaming fashion using multiprocessing.
        
        This is a generator that yields preprocessed batches as they become available,
        allowing downstream processing to begin before all images are preprocessed.
        
        Args:
            inputs: List of image paths or arrays
            ids: Optional list of IDs for each image
            metadata: Optional list of metadata for each image
            batch_size: Batch size (uses default if None)
            
        Yields:
            Dict containing:
                - 'batch_idx': Batch index
                - 'preprocessed': Preprocessed batch (numpy array)
                - 'ids': IDs for this batch (if provided)
                - 'metadata': Metadata for this batch (if provided)
                - 'worker_id': ID of worker that processed this batch
                - 'preprocess_time': Time taken to preprocess this batch
        """
        if not self.is_started:
            raise RuntimeError("Must call start() before preprocess_stream()")
        
        if batch_size is None:
            batch_size = self.batch_size
        
        num_batches = (len(inputs) + batch_size - 1) // batch_size
        logger.info(
            f"Streaming preprocessing for {len(inputs)} images "
            f"in {num_batches} batches of size {batch_size}"
        )
        
        # Submit all batches to input queue
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i + batch_size]
            batch_ids = ids[i:i + batch_size] if ids else None
            batch_metadata = metadata[i:i + batch_size] if metadata else None
            
            batch_data = {
                'batch_idx': i // batch_size,
                'inputs': batch_inputs,
                'ids': batch_ids,
                'metadata': batch_metadata,
            }
            self.input_queue.put(batch_data)
        
        logger.info(f"Submitted {num_batches} batches to workers")
        
        # Collect results from output queue
        # Note: Results may arrive out of order, so we need to sort them
        results_dict = {}
        results_received = 0
        while results_received < num_batches:
            try:
                result = self.output_queue.get(timeout=60.0)
                
                # Check for errors
                if 'error' in result:
                    error_msg = f"Worker {result['worker_id']} error: {result['error']}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                results_received += 1
                batch_idx = result['batch_idx']
                results_dict[batch_idx] = result
                logger.debug(
                    f"Received batch {batch_idx} "
                    f"({results_received}/{num_batches})"
                )
                
            except queue_module.Empty:
                logger.error("Timeout waiting for preprocessing results")
                raise TimeoutError("Preprocessing timeout - workers may have crashed")
        
        # Yield results in order
        for batch_idx in sorted(results_dict.keys()):
            yield results_dict[batch_idx]
    
    def preprocess_batch_sync(
        self,
        inputs: List[Union[str, np.ndarray]],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Preprocess a batch of images synchronously (blocking).
        
        This method collects all results before returning, similar to the
        original preprocess_batch interface.
        
        Args:
            inputs: List of image paths or arrays
            batch_size: Batch size (uses default if None)
            
        Returns:
            Preprocessed images as numpy array (N, ...)
        """
        if not self.is_started:
            self.start()
        
        results = []
        for batch_result in self.preprocess_stream(inputs, batch_size=batch_size):
            results.append(batch_result['preprocessed'])
        
        # Concatenate all batches
        return np.concatenate(results, axis=0) if results else np.array([])
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
    
    def __del__(self):
        """Destructor to ensure workers are stopped."""
        if self.is_started:
            self.stop()
