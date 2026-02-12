"""
Configuration management for the image embedding service.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import yaml
import os
from pathlib import Path


@dataclass
class TritonConfig:
    """Triton Inference Server configuration."""
    url: str = "localhost:8000"
    model_name: str = "embedding_model"
    model_version: str = "1"
    protocol: str = "http"  # http or grpc
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class MilvusConfig:
    """Milvus vector database configuration."""
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "image_embeddings"
    embedding_dim: int = 512
    index_type: str = "IVF_FLAT"  # IVF_FLAT, HNSW, FLAT
    metric_type: str = "L2"  # L2, IP, COSINE
    nlist: int = 128  # for IVF_FLAT
    nprobe: int = 16  # search parameter
    M: int = 16  # for HNSW
    efConstruction: int = 256  # for HNSW


@dataclass
class PreprocessConfig:
    """Image preprocessing configuration."""
    image_size: tuple = (224, 224)
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    batch_size: int = 32
    num_workers: int = 4


@dataclass
class ServiceConfig:
    """Main service configuration."""
    triton: TritonConfig = field(default_factory=TritonConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    log_level: str = "INFO"
    cache_compiled_functions: bool = True
    
    @classmethod
    def from_yaml(cls, path: str) -> "ServiceConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        if 'triton' in data:
            config.triton = TritonConfig(**data['triton'])
        if 'milvus' in data:
            config.milvus = MilvusConfig(**data['milvus'])
        if 'preprocess' in data:
            config.preprocess = PreprocessConfig(**data['preprocess'])
        if 'log_level' in data:
            config.log_level = data['log_level']
        if 'cache_compiled_functions' in data:
            config.cache_compiled_functions = data['cache_compiled_functions']
            
        return config
    
    @classmethod
    def from_env(cls) -> "ServiceConfig":
        """Load configuration from environment variables."""
        config = cls()
        
        # Triton
        if os.getenv('TRITON_URL'):
            config.triton.url = os.getenv('TRITON_URL')
        if os.getenv('TRITON_MODEL_NAME'):
            config.triton.model_name = os.getenv('TRITON_MODEL_NAME')
        if os.getenv('TRITON_MODEL_VERSION'):
            config.triton.model_version = os.getenv('TRITON_MODEL_VERSION')
        if os.getenv('TRITON_PROTOCOL'):
            config.triton.protocol = os.getenv('TRITON_PROTOCOL')
            
        # Milvus
        if os.getenv('MILVUS_HOST'):
            config.milvus.host = os.getenv('MILVUS_HOST')
        if os.getenv('MILVUS_PORT'):
            config.milvus.port = int(os.getenv('MILVUS_PORT'))
        if os.getenv('MILVUS_COLLECTION_NAME'):
            config.milvus.collection_name = os.getenv('MILVUS_COLLECTION_NAME')
        if os.getenv('MILVUS_EMBEDDING_DIM'):
            config.milvus.embedding_dim = int(os.getenv('MILVUS_EMBEDDING_DIM'))
            
        # Logging
        if os.getenv('LOG_LEVEL'):
            config.log_level = os.getenv('LOG_LEVEL')
            
        return config
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        data = {
            'triton': {
                'url': self.triton.url,
                'model_name': self.triton.model_name,
                'model_version': self.triton.model_version,
                'protocol': self.triton.protocol,
                'timeout': self.triton.timeout,
                'max_retries': self.triton.max_retries,
                'retry_delay': self.triton.retry_delay,
            },
            'milvus': {
                'host': self.milvus.host,
                'port': self.milvus.port,
                'collection_name': self.milvus.collection_name,
                'embedding_dim': self.milvus.embedding_dim,
                'index_type': self.milvus.index_type,
                'metric_type': self.milvus.metric_type,
                'nlist': self.milvus.nlist,
                'nprobe': self.milvus.nprobe,
                'M': self.milvus.M,
                'efConstruction': self.milvus.efConstruction,
            },
            'preprocess': {
                'image_size': list(self.preprocess.image_size),
                'mean': list(self.preprocess.mean),
                'std': list(self.preprocess.std),
                'batch_size': self.preprocess.batch_size,
                'num_workers': self.preprocess.num_workers,
            },
            'log_level': self.log_level,
            'cache_compiled_functions': self.cache_compiled_functions,
        }
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)


# Default configuration instance
default_config = ServiceConfig()
