# Async Pipeline Architecture / 异步管道架构

## English Version

### Overview

The async pipeline architecture optimizes throughput by decoupling preprocessing, embedding generation, and database insertion operations. This prevents the GPU from waiting for database operations, significantly improving overall throughput.

### Architecture

```
┌─────────────────┐
│  Main Thread    │  JAX preprocessing in batches
│  (preprocessing)│  (must run in main thread)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Producer       │  Queue management
│  Thread         │  (feeds preprocessed data)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Embedding      │  GPU inference
│  Worker Pool    │  (Triton inference)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Queue          │  Buffering embeddings
│  (async)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Milvus Async   │  Batch insertion
│  Inserter       │  (async database ops)
└─────────────────┘
```

**Note:** JAX preprocessing runs in the main thread to avoid thread-switching errors. The producer thread only manages queue operations.

### Problem Statement

**Before (Synchronous Pipeline):**
```
preprocess → embed → insert → preprocess → embed → insert
     GPU idle ↑           ↑ GPU idle
```

The GPU waits for:
- Data preprocessing to complete
- Database insertion to complete

**After (Async Pipeline):**
```
Main Thread (JAX preprocess) ──→ Producer Thread (queue) ──→ Embedding Workers (GPU) ──→ Queue ──→ Milvus Inserter
```

The GPU never waits:
- JAX preprocessing happens in main thread (avoids thread-switching errors)
- Queue management happens in producer thread
- Database operations happen asynchronously in background

### Key Benefits

1. **Higher Throughput**: GPU continuously processes without waiting for I/O
2. **Better Resource Utilization**: CPU, GPU, and database work in parallel
3. **Improved Scalability**: Can handle larger batch sizes with controlled memory
4. **Flexible Configuration**: Tune worker counts and batch sizes for your workload

### Usage

#### Basic Usage

```python
from src.pipeline import ImageEmbeddingPipeline
from src.config import ServiceConfig

config = ServiceConfig.from_yaml('configs/config.yaml')

with ImageEmbeddingPipeline(config) as pipeline:
    # Create collection
    pipeline.create_collection("my_images", dim=512)
    
    # Use async pipeline for better throughput
    ids = pipeline.insert_images_async(
        inputs=image_paths,
        ids=image_ids,
        metadata=metadata,
        collection_name="my_images",
    )
```

#### Advanced Configuration

```python
from src.config import ServiceConfig

config = ServiceConfig()

# Configure async pipeline
config.async_pipeline.preprocess_workers = 4    # More workers for CPU-bound preprocessing
config.async_pipeline.embedding_workers = 1     # Usually 1 for single GPU
config.async_pipeline.insert_batch_size = 200   # Larger batches for database
config.async_pipeline.queue_maxsize = 150       # Buffer size between stages

pipeline = ImageEmbeddingPipeline(config)

# Override per-call
ids = pipeline.insert_images_async(
    inputs=image_paths,
    ids=image_ids,
    preprocess_workers=2,     # Override config
    embedding_workers=1,
    insert_batch_size=100,
)
```

#### YAML Configuration

```yaml
async_pipeline:
  preprocess_workers: 2      # Number of preprocessing threads
  embedding_workers: 1       # Number of embedding workers (usually 1 for GPU)
  insert_batch_size: 100     # Batch size for Milvus insertion
  queue_maxsize: 100         # Maximum queue size

preprocess:
  batch_size: 32             # Batch size for preprocessing/embedding
```

### Performance Comparison

Run the benchmark to compare sync vs async:

```bash
python examples/async_batch_processing.py
```

Expected improvements:
- **20-50% faster** for typical workloads
- **Higher speedup** when database operations are slow
- **Better GPU utilization** with continuous processing

### Configuration Guide

#### Preprocess Workers

Number of threads for parallel image loading and preprocessing.

- **Default**: 2
- **Recommendation**: 2-4 for local files, 4-8 for URLs
- **Trade-off**: More workers = faster preprocessing but more memory

#### Embedding Workers

Number of workers for GPU inference.

- **Default**: 1
- **Recommendation**: 1 for single GPU, match GPU count for multi-GPU
- **Trade-off**: More workers only help with multiple GPUs

#### Insert Batch Size

Number of embeddings to accumulate before inserting to Milvus.

- **Default**: 100
- **Recommendation**: 50-200 depending on embedding dimension and memory
- **Trade-off**: Larger batches = fewer database calls but more memory

#### Queue Maxsize

Maximum number of items in the queue between stages.

- **Default**: 100
- **Recommendation**: 50-200 depending on available memory
- **Trade-off**: Larger queue = more buffering but more memory

### Best Practices

1. **Start with defaults**: The default configuration works well for most cases
2. **Monitor GPU utilization**: Use `nvidia-smi` to ensure GPU stays busy
3. **Tune for your workload**: 
   - More preprocess workers for slow I/O (network, disk)
   - Larger insert batches for high-latency databases
   - Larger queues for bursty workloads
4. **Memory constraints**: Reduce batch sizes and queue sizes if running out of memory

### Comparison with Sync Pipeline

| Feature | Sync Pipeline | Async Pipeline |
|---------|--------------|----------------|
| Throughput | Baseline | 20-50% faster |
| GPU Utilization | Lower (waits for I/O) | Higher (continuous) |
| Memory Usage | Lower | Higher (buffering) |
| Complexity | Simple | Moderate |
| Best For | Small datasets | Large datasets |

### When to Use

**Use Async Pipeline When:**
- Processing large datasets (1000+ images)
- Database insertion is slow
- Maximum throughput is important
- You have sufficient memory for buffering

**Use Sync Pipeline When:**
- Processing small datasets (<100 images)
- Memory is constrained
- Simplicity is more important than throughput
- Real-time processing with minimal latency

---

## 中文版本

### 概述

异步管道架构通过解耦预处理、嵌入生成和数据库插入操作来优化吞吐量。这可以防止 GPU 等待数据库操作，显著提高整体吞吐量。

### 架构

```
┌─────────────────┐
│  生产者线程      │  批量预处理
│  Producer       │  (JAX 预处理)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  嵌入工作池      │  GPU 推理
│  Embedding Pool │  (Triton 推理)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  队列           │  缓冲嵌入向量
│  Queue          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Milvus 异步    │  批量插入
│  插入器         │  (异步数据库操作)
└─────────────────┘
```

### 问题陈述

**之前（同步管道）：**
```
预处理 → 嵌入 → 插入 → 预处理 → 嵌入 → 插入
  GPU 空闲 ↑        ↑ GPU 空闲
```

GPU 等待：
- 数据预处理完成
- 数据库插入完成

**之后（异步管道）：**
```
生产者（预处理）──┐
               ├──→ 嵌入工作器（GPU）──→ 队列 ──→ Milvus 插入器
生产者（预处理）──┘
```

GPU 永不等待：
- 预处理在并行线程中进行
- 数据库操作在后台异步进行

### 主要优势

1. **更高的吞吐量**：GPU 持续处理，无需等待 I/O
2. **更好的资源利用**：CPU、GPU 和数据库并行工作
3. **改进的可扩展性**：可以通过受控内存处理更大的批次
4. **灵活的配置**：根据工作负载调整工作器数量和批次大小

### 使用方法

#### 基本使用

```python
from src.pipeline import ImageEmbeddingPipeline
from src.config import ServiceConfig

config = ServiceConfig.from_yaml('configs/config.yaml')

with ImageEmbeddingPipeline(config) as pipeline:
    # 创建集合
    pipeline.create_collection("my_images", dim=512)
    
    # 使用异步管道以获得更好的吞吐量
    ids = pipeline.insert_images_async(
        inputs=image_paths,
        ids=image_ids,
        metadata=metadata,
        collection_name="my_images",
    )
```

### 性能比较

运行基准测试来比较同步与异步：

```bash
python examples/async_batch_processing.py
```

预期改进：
- 对于典型工作负载**快 20-50%**
- 当数据库操作较慢时**加速更高**
- 通过持续处理实现**更好的 GPU 利用率**

### 最佳实践

1. **从默认值开始**：默认配置适用于大多数情况
2. **监控 GPU 利用率**：使用 `nvidia-smi` 确保 GPU 保持忙碌
3. **针对您的工作负载进行调整**：
   - 对于慢速 I/O（网络、磁盘），增加预处理工作器
   - 对于高延迟数据库，增加插入批次大小
   - 对于突发工作负载，增加队列大小
4. **内存约束**：如果内存不足，减少批次大小和队列大小
