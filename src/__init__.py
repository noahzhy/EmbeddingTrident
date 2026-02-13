"""
JAX + Triton + Milvus Image Embedding Service

A production-ready image embedding pipeline with:
- JAX-accelerated preprocessing (jit + vmap)
- Triton Inference Server integration
- Milvus vector database support
"""

__version__ = "0.1.0"

# Export main components
from .base_preprocessor import ImagePreprocessor
from .preprocess_jax import JAXImagePreprocessor
from .pipeline import ImageEmbeddingPipeline
from .config import ServiceConfig, PreprocessConfig, TritonConfig, MilvusConfig

__all__ = [
    "ImagePreprocessor",
    "JAXImagePreprocessor",
    "ImageEmbeddingPipeline",
    "ServiceConfig",
    "PreprocessConfig",
    "TritonConfig",
    "MilvusConfig",
]
