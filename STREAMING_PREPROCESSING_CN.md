# Streaming Multiprocessing Preprocessing

## 流式多进程预处理 (Streaming Multiprocessing Preprocessing)

### 概述 (Overview)

This implementation adds streaming multiprocessing preprocessing to accelerate overall efficiency by parallelizing preprocessing across multiple CPU cores.

本实现添加了流式多进程预处理，通过在多个 CPU 核心上并行预处理来加速整体效率。

### 实现细节 (Implementation Details)

#### 新增文件 (New Files)
- `src/streaming_preprocessor.py`: `StreamingMultiprocessPreprocessor` 类
- `tests/test_streaming_preprocessor.py`: 综合测试套件
- `tests/benchmark_streaming.py`: 性能基准测试

#### 修改文件 (Modified Files)
- `src/pipeline.py`: 添加了 `insert_images_streaming()` 方法

### 架构 (Architecture)

```
[Preprocessing Workers (multiprocessing)] -> Preprocessed Queue -> 
[Embedding Workers (gevent)] -> Embedding Queue -> 
[Milvus Inserter (gevent)]
```

每个预处理工作进程 (Each preprocessing worker):
1. 从输入队列接收批次数据 (Receives batch data from input queue)
2. 使用 ThreadPoolExecutor 加载图像 (Loads images using ThreadPoolExecutor)
3. 使用 JAX (jit + vmap) 预处理 (Preprocesses using JAX)
4. 将预处理的批次发送到输出队列 (Sends preprocessed batch to output queue)

### 使用场景 (Use Cases)

#### 最适合 (Best For):
- **大规模批处理** (Large-scale batch processing): 数千张图像以上 (Thousands of images or more)
- **持续处理** (Continuous processing): 长时间运行的服务 (Long-running services)
- **I/O 密集型** (I/O-bound): 从慢速存储或网络加载图像 (Loading images from slow storage or network)

#### 不适合 (Not Ideal For):
- **小批量** (Small batches): 少于100张图像 (Less than 100 images)
- **一次性处理** (One-off processing): 进程启动开销很大 (Process startup overhead is significant)
- **已经在内存中的数据** (Data already in memory): 无 I/O 瓶颈 (No I/O bottleneck)

### 性能特征 (Performance Characteristics)

#### 开销 (Overhead):
- 进程启动: ~0.5-1秒 (Process startup: ~0.5-1s)
- JAX 编译 (每个进程): ~0.3-0.5秒 (JAX compilation per process: ~0.3-0.5s)
- 总启动开销: ~1-2秒 (Total startup overhead: ~1-2s)

#### 吞吐量 (Throughput):
- 对于大批量 (>1000张图像), 预期加速 2-4x (For large batches (>1000 images), expect 2-4x speedup)
- 随着批量大小增加，加速效果更明显 (Speedup increases with batch size)
- CPU 核心越多，加速效果越好 (More CPU cores = better speedup)

### 使用示例 (Usage Examples)

#### 1. 基本使用 (Basic Usage)

```python
from src.streaming_preprocessor import StreamingMultiprocessPreprocessor

# 创建流式预处理器 (Create streaming preprocessor)
preprocessor = StreamingMultiprocessPreprocessor(
    num_workers=4,
    batch_size=32,
    image_size=(224, 224),
)

# 使用上下文管理器 (Use context manager)
with preprocessor:
    for batch_result in preprocessor.preprocess_stream(image_paths, batch_size=32):
        preprocessed = batch_result['preprocessed']
        ids = batch_result['ids']
        # 处理预处理的批次... (Process preprocessed batch...)
```

#### 2. 与管道集成 (Integration with Pipeline)

```python
from src.pipeline import ImageEmbeddingPipeline
from src.config import ServiceConfig

config = ServiceConfig.from_yaml('configs/config.yaml')
pipeline = ImageEmbeddingPipeline(config)

# 使用流式多进程预处理 (Use streaming multiprocessing preprocessing)
inserted_ids = pipeline.insert_images_streaming(
    inputs=image_paths,
    ids=image_ids,
    metadata=metadata,
    batch_size=32,
    num_preprocess_workers=4,
)
```

#### 3. 同步批处理 (Synchronous Batch Processing)

```python
# 如果需要阻塞直到所有结果准备好 (If you need to block until all results are ready)
with preprocessor:
    all_preprocessed = preprocessor.preprocess_batch_sync(image_paths, batch_size=32)
```

### 配置建议 (Configuration Recommendations)

#### num_workers (工作进程数)
- 默认: CPU 核心数 (Default: CPU count)
- 建议: CPU 核心数 或 CPU 核心数 - 1 (Recommended: CPU count or CPU count - 1)
- 太多会增加开销 (Too many increases overhead)

#### batch_size (批次大小)
- 默认: 32
- 建议: 16-64 取决于图像大小 (Recommended: 16-64 depending on image size)
- 更大的批次可以更好地利用 JAX 的 vmap (Larger batches utilize JAX's vmap better)

#### queue_maxsize (队列大小)
- 默认: 10
- 建议: 5-20
- 更大的队列使用更多内存但可以更好地平滑处理 (Larger queues use more memory but smooth out processing)

### 测试结果 (Test Results)

所有测试通过 ✓ (All tests passing ✓):
- ✓ 基本流式预处理功能 (Basic streaming preprocessor functionality)
- ✓ 带 IDs 和元数据的流式处理 (Streaming with IDs and metadata)
- ✓ 同步批处理 (Synchronous batch processing)
- ✓ 错误处理 (Error handling)
- ✓ 自定义预处理器 (Custom preprocessor)

### 向后兼容性 (Backward Compatibility)

- 原有的 `insert_images()` 和 `insert_images_async()` 方法保持不变 (Original methods remain unchanged)
- 新方法 `insert_images_streaming()` 是可选的 (New `insert_images_streaming()` method is optional)
- 所有现有代码继续正常工作 (All existing code continues to work)

### 未来改进 (Future Improvements)

1. **进程池预热** (Process pool warmup): 预先启动工作进程以减少延迟 (Pre-start workers to reduce latency)
2. **智能批处理** (Smart batching): 根据负载动态调整批次大小 (Dynamically adjust batch size based on load)
3. **GPU 支持** (GPU support): 在不同进程中使用不同的 GPU (Use different GPUs in different processes)
4. **内存优化** (Memory optimization): 共享内存用于大批次 (Shared memory for large batches)

### 总结 (Summary)

流式多进程预处理提供了一种有效的方式来加速大规模图像预处理，通过在多个 CPU 核心上并行处理来提高吞吐量。它最适合需要处理大量图像的场景，而对于小批量处理，传统的方法可能更高效。

Streaming multiprocessing preprocessing provides an effective way to accelerate large-scale image preprocessing by parallelizing across multiple CPU cores. It's best suited for scenarios with large batches of images, while traditional methods may be more efficient for small batches.
