# Performance Optimization Guide

This document provides tips and best practices for optimizing the image embedding service performance.

## JAX Preprocessing Optimization

### 1. JIT Compilation

The preprocessing functions are JIT-compiled by default. Make sure to enable caching:

```python
preprocessor = JAXImagePreprocessor(cache_compiled=True)
```

### 2. Batch Size Tuning

Optimal batch size depends on your hardware:

- **CPU**: 8-16
- **Single GPU**: 32-64
- **Multi-GPU**: 128-256

### 3. Image Size

Smaller images process faster:

- 224x224: Standard, good balance
- 384x384: Better quality, slower
- 512x512: High quality, slowest

## Triton Inference Optimization

### 1. Dynamic Batching

Enable in `config.pbtxt`:

```protobuf
dynamic_batching {
  max_queue_delay_microseconds: 100
}
```

### 2. Model Optimization

Use optimized model formats:
- ONNX with ONNX Runtime
- TensorRT for NVIDIA GPUs

### 3. Protocol Selection

- **HTTP**: Easier to use, good for most cases
- **gRPC**: Lower latency, better for high throughput

## Milvus Optimization

### 1. Index Selection

**IVF_FLAT** (Default)
- Fast training, good recall

**HNSW**
- Best search quality, higher memory usage

**FLAT**
- Brute force, only for small datasets

### 2. Batch Insert

Always batch inserts for better throughput:

```python
pipeline.insert_images(images, ids, batch_size=64)
```

## Benchmark Results

Expected performance on different hardware:

### Single GPU (NVIDIA RTX 3090)
- Preprocessing: ~50 images/sec
- Inference: ~120 images/sec
- Insert: ~8000 vectors/sec
- Search: ~25ms (TopK=10)

## Quick Wins

1. **Enable JIT caching**: `cache_compiled=True`
2. **Use optimal batch size**: 32-64 for single GPU
3. **Enable dynamic batching**: In Triton config
4. **Use gRPC protocol**: `protocol="grpc"`
5. **Batch inserts**: Always insert in batches
