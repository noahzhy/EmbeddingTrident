# 问题修复总结

本次提交解决了两个问题：

## 1. 修复 custom_preprocessor.py 文件

**问题**: 文件中有重复的代码，导致语法错误

**解决方案**:
- 删除了 209 行之后的重复代码（236 行）
- 文件从 445 行减少到 209 行
- 现在文件可以正常导入和使用

**验证**:
```bash
python examples/custom_preprocessor.py
```

## 2. 添加 GPU 索引支持到 milvus_client.py

**新增的 GPU 索引类型**:

### GPU_CAGRA
基于图的 GPU 加速索引，提供最佳的搜索性能

**参数**:
- `intermediate_graph_degree`: 中间图度数（默认：64）
- `graph_degree`: 最终图度数（默认：32）
- `itopk_size`: 搜索时的内部 top-k 大小（默认：64）
- `search_width`: 搜索宽度（默认：4）
- `min_iterations`: 最小迭代次数（默认：0）
- `max_iterations`: 最大迭代次数（默认：0）
- `team_size`: 并行搜索的团队大小（默认：0）

**使用示例**:
```python
from src.milvus_client import MilvusClient

client = MilvusClient(
    index_type="GPU_CAGRA",
    graph_degree=32,
    itopk_size=64,
    search_width=4
)
```

### GPU_IVF_FLAT
GPU 加速的 IVF 索引，使用扁平存储

**参数**:
- `nlist`: 聚类单元数（默认：128）
- `nprobe`: 搜索时查询的单元数（默认：16）

**使用示例**:
```python
client = MilvusClient(
    index_type="GPU_IVF_FLAT",
    nlist=128,
    nprobe=16
)
```

### GPU_IVF_PQ
GPU 加速的 IVF 索引，使用乘积量化压缩

**参数**:
- `nlist`: 聚类单元数（默认：128）
- `nprobe`: 搜索时查询的单元数（默认：16）
- `m`: 子量化器数量（默认：8，可配置）
- `nbits`: 每个子量化器的位数（默认：8，可配置）

**使用示例**:
```python
client = MilvusClient(
    index_type="GPU_IVF_PQ",
    nlist=256,
    nprobe=32,
    m=16,      # 自定义子量化器数量
    nbits=16   # 自定义位数
)
```

### GPU_BRUTE_FORCE
GPU 加速的暴力搜索（穷举搜索）

**参数**: 无（仅使用 metric_type）

**使用示例**:
```python
client = MilvusClient(
    index_type="GPU_BRUTE_FORCE",
    metric_type="IP"
)
```

## 配置文件示例

在 `config.yaml` 中配置 GPU 索引：

```yaml
milvus:
  host: localhost
  port: 19530
  collection_name: image_embeddings
  embedding_dim: 512
  index_type: GPU_CAGRA  # 或 GPU_IVF_FLAT, GPU_IVF_PQ, GPU_BRUTE_FORCE
  metric_type: L2
  
  # GPU_CAGRA 参数
  intermediate_graph_degree: 64
  graph_degree: 32
  itopk_size: 64
  search_width: 4
  
  # GPU_IVF_PQ 参数（如果使用 GPU_IVF_PQ）
  nlist: 128
  nprobe: 16
  m: 8
  nbits: 8
```

## 性能对比

与 CPU 索引相比的典型性能提升：

| 索引类型 | 构建速度 | 搜索速度 | 内存使用 | 召回率 |
|---------|---------|---------|---------|--------|
| GPU_CAGRA | ~2倍 | ~10-30倍 | 高 | 很高(>95%) |
| GPU_IVF_FLAT | ~3倍 | ~5-15倍 | 中 | 高(>90%) |
| GPU_IVF_PQ | ~3倍 | ~3-10倍 | 低 | 中(>80%) |
| GPU_BRUTE_FORCE | 无需构建 | ~5-10倍 | 低 | 100% |

## 测试

运行 GPU 索引测试：
```bash
python tests/test_gpu_indexes.py
```

所有测试通过 ✓

## 文档

详细文档请参阅：
- [GPU 索引支持指南](../docs/GPU_INDEX_SUPPORT.md)（英文）
- [测试文件](../tests/test_gpu_indexes.py)

## 兼容性

- 完全向后兼容现有的 CPU 索引
- 可以通过配置文件或代码轻松切换索引类型
- 所有 GPU 参数都有合理的默认值

## 要求

使用 GPU 索引需要：
1. NVIDIA GPU（支持 CUDA，计算能力 ≥ 7.0）
2. 启用 GPU 支持的 Milvus 服务器
3. 足够的 GPU 内存（取决于数据集大小和索引类型）
