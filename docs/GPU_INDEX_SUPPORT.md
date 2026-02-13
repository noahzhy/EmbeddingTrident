# GPU Index Support in Milvus Client

This document describes the GPU-accelerated index types supported in the Milvus client.

## Overview

The Milvus client now supports GPU-accelerated indexes for faster vector similarity search on systems with NVIDIA GPUs. These indexes provide significant performance improvements over CPU-based indexes.

## Supported GPU Index Types

### 1. GPU_CAGRA

CAGRA (Cuda Anns GRAph-based) is a graph-based GPU index that provides excellent search performance.

**Use Case**: Best for high-throughput, low-latency search scenarios with sufficient GPU memory.

**Parameters**:
- `intermediate_graph_degree` (default: 64): Intermediate graph degree during index construction
- `graph_degree` (default: 32): Final graph degree
- `itopk_size` (default: 64): Internal top-k size for search
- `search_width` (default: 4): Search width parameter
- `min_iterations` (default: 0): Minimum search iterations
- `max_iterations` (default: 0): Maximum search iterations
- `team_size` (default: 0): Team size for parallel search

**Example**:
```python
from src.milvus_client import MilvusClient

client = MilvusClient(
    host="localhost",
    port=19530,
    index_type="GPU_CAGRA",
    metric_type="L2",
    intermediate_graph_degree=64,
    graph_degree=32,
    itopk_size=64,
    search_width=4,
)
```

### 2. GPU_IVF_FLAT

GPU-accelerated IVF (Inverted File) index with flat (uncompressed) vector storage.

**Use Case**: Good balance between speed and accuracy for medium-to-large datasets with GPU acceleration.

**Parameters**:
- `nlist` (default: 128): Number of cluster units
- `nprobe` (default: 16): Number of units to query during search

**Example**:
```python
client = MilvusClient(
    index_type="GPU_IVF_FLAT",
    metric_type="IP",
    nlist=128,
    nprobe=16,
)
```

### 3. GPU_IVF_PQ

GPU-accelerated IVF index with Product Quantization for memory efficiency.

**Use Case**: Large-scale datasets where memory is a constraint. Trades some accuracy for reduced memory usage.

**Parameters**:
- `nlist` (default: 128): Number of cluster units
- `nprobe` (default: 16): Number of units to query during search
- `m` (default: 8): Number of subquantizers (automatically set)
- `nbits` (default: 8): Bits per subquantizer (automatically set)

**Example**:
```python
client = MilvusClient(
    index_type="GPU_IVF_PQ",
    metric_type="L2",
    nlist=256,
    nprobe=32,
)
```

### 4. GPU_BRUTE_FORCE

GPU-accelerated brute-force search (exhaustive search).

**Use Case**: Small datasets or when 100% recall is required. No index building required.

**Parameters**: None (uses metric_type only)

**Example**:
```python
client = MilvusClient(
    index_type="GPU_BRUTE_FORCE",
    metric_type="IP",
)
```

## Complete Usage Example

```python
from src.pipeline import ImageEmbeddingPipeline
from src.config import ServiceConfig

# Configure for GPU_CAGRA index
config = ServiceConfig()
config.milvus.index_type = "GPU_CAGRA"
config.milvus.metric_type = "L2"

# Create pipeline
pipeline = ImageEmbeddingPipeline(config)

# Create collection (will use GPU_CAGRA index)
pipeline.create_collection(
    name="gpu_images",
    dim=512,
    description="GPU-accelerated image embeddings"
)

# Insert images
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
ids = ["img_1", "img_2", "img_3"]
pipeline.insert_images(
    inputs=image_paths,
    ids=ids,
    collection_name="gpu_images"
)

# Search (automatically uses GPU-accelerated search)
results = pipeline.search_images(
    query_input="query.jpg",
    topk=10,
    collection_name="gpu_images"
)
```

## Performance Comparison

Typical performance improvements with GPU indexes (compared to CPU indexes):

| Index Type | Build Time | Search Speed | Memory Usage | Recall |
|------------|-----------|--------------|--------------|---------|
| GPU_CAGRA | ~2x faster | ~10-30x faster | High | Very High (>95%) |
| GPU_IVF_FLAT | ~3x faster | ~5-15x faster | Medium | High (>90%) |
| GPU_IVF_PQ | ~3x faster | ~3-10x faster | Low | Medium (>80%) |
| GPU_BRUTE_FORCE | N/A | ~5-10x faster | Low | 100% |

*Note: Actual performance depends on dataset size, GPU model, and configuration.*

## Configuration via YAML

You can configure GPU indexes in your `config.yaml`:

```yaml
milvus:
  host: localhost
  port: 19530
  collection_name: image_embeddings
  embedding_dim: 512
  index_type: GPU_CAGRA  # or GPU_IVF_FLAT, GPU_IVF_PQ, GPU_BRUTE_FORCE
  metric_type: L2
  # GPU_CAGRA specific
  intermediate_graph_degree: 64
  graph_degree: 32
  itopk_size: 64
  search_width: 4
  min_iterations: 0
  max_iterations: 0
  team_size: 0
  # GPU_IVF_PQ specific (if using GPU_IVF_PQ)
  nlist: 128
  nprobe: 16
  m: 8  # Number of subquantizers
  nbits: 8  # Bits per subquantizer
```

## Requirements

To use GPU indexes, you need:

1. **NVIDIA GPU**: With CUDA support (Compute Capability ≥ 7.0)
2. **Milvus GPU Version**: Milvus server with GPU support enabled
3. **Sufficient GPU Memory**: Depends on dataset size and index type

## Best Practices

1. **Choose the Right Index**:
   - `GPU_CAGRA`: Best for high-performance requirements
   - `GPU_IVF_FLAT`: Good balance for most use cases
   - `GPU_IVF_PQ`: Use when memory is limited
   - `GPU_BRUTE_FORCE`: Small datasets or when perfect recall is needed

2. **Tune Parameters**:
   - For GPU_CAGRA: Increase `graph_degree` for better recall, decrease for speed
   - For IVF indexes: Increase `nlist` for larger datasets, increase `nprobe` for better recall

3. **Monitor GPU Memory**:
   - GPU indexes require more GPU memory than CPU indexes
   - Use `nvidia-smi` to monitor GPU memory usage

4. **Batch Operations**:
   - Batch inserts and searches for better throughput
   - GPU indexes excel at batch processing

## Troubleshooting

**Issue**: "GPU index not supported"
- **Solution**: Ensure Milvus server has GPU support enabled and CUDA is properly installed

**Issue**: Out of GPU memory
- **Solution**: 
  - Use GPU_IVF_PQ instead of GPU_IVF_FLAT
  - Reduce batch sizes
  - Use smaller embedding dimensions

**Issue**: Slower than expected
- **Solution**:
  - Ensure data is loaded into GPU memory
  - Increase batch sizes for search
  - Tune index parameters (nprobe, search_width, etc.)

## Migration from CPU Indexes

To migrate from CPU to GPU indexes:

```python
# Old CPU configuration
client = MilvusClient(
    index_type="IVF_FLAT",
    nlist=128,
    nprobe=16,
)

# New GPU configuration (minimal change)
client = MilvusClient(
    index_type="GPU_IVF_FLAT",  # Just add "GPU_" prefix
    nlist=128,
    nprobe=16,
)
```

For better performance, consider switching to GPU_CAGRA:

```python
client = MilvusClient(
    index_type="GPU_CAGRA",
    graph_degree=32,
    itopk_size=64,
)
```

## References

- [Milvus Documentation](https://milvus.io/docs/)
- [Milvus GPU Index Guide](https://milvus.io/docs/index.md)
- [CAGRA Algorithm Paper](https://arxiv.org/abs/2308.15136)
- [Product Quantization](https://ieeexplore.ieee.org/document/5432202)
