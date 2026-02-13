"""
Triton Inference Server client for embedding extraction.
"""

import numpy as np
import jax.numpy as jnp
from jax import jit
from typing import List, Optional, Dict, Any
import time
from loguru import logger

try:
    import tritonclient.http as httpclient
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import InferenceServerException
except ImportError:
    logger.warning("tritonclient not installed. Install with: pip install tritonclient[all]")
    httpclient = None
    grpcclient = None
    InferenceServerException = Exception


class TritonClient:
    """
    High-performance Triton Inference Server client.
    
    Features:
    - HTTP and gRPC protocol support
    - Batch inference optimization
    - Automatic retry on transient failures
    - L2 normalization of embeddings
    - Async inference support
    """
    
    def __init__(
        self,
        url: str = "localhost:8000",
        model_name: str = "embedding_model",
        model_version: str = "1",
        protocol: str = "http",
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        input_name: str = "input",
        output_name: str = "output",
    ):
        """
        Initialize Triton client.
        
        Args:
            url: Triton server URL (host:port)
            model_name: Model name in Triton
            model_version: Model version
            protocol: 'http' or 'grpc'
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
            input_name: Default input tensor name in model
            output_name: Default output tensor name in model
        """
        self.url = url
        self.model_name = model_name
        self.model_version = model_version
        self.protocol = protocol.lower()
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.input_name = input_name
        self.output_name = output_name
        
        # Initialize client
        self.client = self._create_client()
        
        # Check server health
        self._check_server_health()
        
        # Get model metadata
        self._get_model_metadata()
    
    def _create_client(self):
        """Create Triton client based on protocol."""
        if self.protocol == "http":
            if httpclient is None:
                raise ImportError("tritonclient.http not available")
            return httpclient.InferenceServerClient(
                url=self.url,
                connection_timeout=self.timeout,
                network_timeout=self.timeout,
            )
        elif self.protocol == "grpc":
            if grpcclient is None:
                raise ImportError("tritonclient.grpc not available")
            return grpcclient.InferenceServerClient(url=self.url)
        else:
            raise ValueError(f"Invalid protocol: {self.protocol}")
    
    def _check_server_health(self) -> None:
        """Check if Triton server is healthy."""
        try:
            if self.client.is_server_live():
                logger.info(f"Triton server at {self.url} is live")
            else:
                logger.warning(f"Triton server at {self.url} is not responding")
        except Exception as e:
            logger.error(f"Failed to connect to Triton server: {e}")
            raise
    
    def _get_model_metadata(self) -> None:
        """Get and validate model metadata."""
        try:
            self.model_metadata = self.client.get_model_metadata(
                model_name=self.model_name,
                model_version=self.model_version,
            )
            logger.info(f"Loaded model metadata for {self.model_name} (version {self.model_version})")
        except Exception as e:
            logger.warning(f"Could not get model metadata: {e}")
            self.model_metadata = None
    
    @staticmethod
    @jit
    def _l2_normalize_jax(embeddings: jnp.ndarray) -> jnp.ndarray:
        """
        JIT-compiled L2 normalization.
        
        Args:
            embeddings: Input embeddings (B, D)
            
        Returns:
            L2-normalized embeddings
        """
        norms = jnp.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = jnp.where(norms == 0, 1.0, norms)
        return embeddings / norms
    
    def l2_normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        L2 normalize embeddings.
        
        Args:
            embeddings: Input embeddings (B, D)
            
        Returns:
            Normalized embeddings
        """
        jax_embeddings = jnp.array(embeddings)
        normalized = self._l2_normalize_jax(jax_embeddings)
        return np.array(normalized)
    
    def infer(
        self,
        inputs: np.ndarray,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Run inference on Triton server.
        
        Args:
            inputs: Input tensor (B, H, W, C) or (B, C, H, W)
            input_name: Input tensor name in model (uses default from config if None)
            output_name: Output tensor name in model (uses default from config if None)
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            Embedding vectors (B, D)
        """
        # Use instance defaults if not provided
        if input_name is None:
            input_name = self.input_name
        if output_name is None:
            output_name = self.output_name
            
        # Ensure inputs are float32
        inputs = inputs.astype(np.float32)
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                # Create input object
                if self.protocol == "http":
                    triton_input = httpclient.InferInput(
                        input_name,
                        inputs.shape,
                        "FP32",
                    )
                    triton_input.set_data_from_numpy(inputs)
                    
                    # Create output object
                    triton_output = httpclient.InferRequestedOutput(output_name)
                    
                    # Run inference
                    response = self.client.infer(
                        model_name=self.model_name,
                        model_version=self.model_version,
                        inputs=[triton_input],
                        outputs=[triton_output],
                    )
                    
                    # Get output
                    embeddings = response.as_numpy(output_name)
                    
                else:  # grpc
                    triton_input = grpcclient.InferInput(
                        input_name,
                        inputs.shape,
                        "FP32",
                    )
                    triton_input.set_data_from_numpy(inputs)
                    
                    # Create output object
                    triton_output = grpcclient.InferRequestedOutput(output_name)
                    
                    # Run inference
                    response = self.client.infer(
                        model_name=self.model_name,
                        model_version=self.model_version,
                        inputs=[triton_input],
                        outputs=[triton_output],
                    )
                    
                    # Get output
                    embeddings = response.as_numpy(output_name)
                
                inference_time = time.time() - start_time
                logger.debug(
                    f"Inference completed in {inference_time:.3f}s "
                    f"(batch_size={inputs.shape[0]})"
                )
                
                # Normalize if requested
                if normalize:
                    embeddings = self.l2_normalize(embeddings)
                
                return embeddings
                
            except InferenceServerException as e:
                logger.error(f"Inference failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise
            except Exception as e:
                logger.error(f"Unexpected error during inference: {e}")
                raise
    
    def infer_batch(
        self,
        inputs_list: List[np.ndarray],
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 32,
    ) -> List[np.ndarray]:
        """
        Run inference on multiple batches.
        
        Args:
            inputs_list: List of input tensors
            input_name: Input tensor name (uses default from config if None)
            output_name: Output tensor name (uses default from config if None)
            normalize: Whether to normalize embeddings
            batch_size: Maximum batch size
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(inputs_list), batch_size):
            batch = inputs_list[i:i + batch_size]
            
            # Stack batch
            if len(batch) == 1:
                batch_inputs = batch[0]
            else:
                batch_inputs = np.concatenate(batch, axis=0)
            
            # Infer
            embeddings = self.infer(
                batch_inputs,
                input_name=input_name,
                output_name=output_name,
                normalize=normalize,
            )
            
            all_embeddings.append(embeddings)
        
        return all_embeddings
    
    def async_infer(
        self,
        inputs: np.ndarray,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
        callback: Optional[callable] = None,
    ) -> Any:
        """
        Run async inference (HTTP only).
        
        Args:
            inputs: Input tensor
            input_name: Input tensor name (uses default from config if None)
            output_name: Output tensor name (uses default from config if None)
            callback: Callback function for async result
            
        Returns:
            Async result handle
        """
        if self.protocol != "http":
            raise NotImplementedError("Async inference only supported for HTTP protocol")
        
        # Use instance defaults if not provided
        if input_name is None:
            input_name = self.input_name
        if output_name is None:
            output_name = self.output_name
            
        # Ensure inputs are float32
        inputs = inputs.astype(np.float32)
        
        # Create input
        triton_input = httpclient.InferInput(input_name, inputs.shape, "FP32")
        triton_input.set_data_from_numpy(inputs)
        
        # Create output
        triton_output = httpclient.InferRequestedOutput(output_name)
        
        # Run async inference
        result = self.client.async_infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=[triton_input],
            outputs=[triton_output],
            callback=callback,
        )
        
        return result
    
    def close(self) -> None:
        """Close the client connection."""
        try:
            self.client.close()
            logger.info("Triton client connection closed")
        except Exception as e:
            logger.warning(f"Error closing client: {e}")
