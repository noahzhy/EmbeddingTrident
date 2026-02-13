# Embedding Trident
JAX + Triton + Milvus Image Embedding Service

A **production-ready image embedding service** combining:
- **JAX** for high-performance preprocessing (jit + vmap)
- **Triton Inference Server** for optimized model serving
- **Milvus** for scalable vector storage and search

## 🚀 Features

- ✅ **High-Performance Pipeline**: JIT-compiled JAX preprocessing with vectorized batch processing
- ✅ **GPU Acceleration**: Optional GPU support for JAX preprocessing (4-5x speedup)
- ✅ **Custom Preprocessors**: Extensible preprocessing interface - bring your own preprocessing logic
- ✅ **Flexible Input**: Support for local files and remote URLs
- ✅ **Batch Processing**: Optimized batching throughout the pipeline
- ✅ **Vector Database**: Milvus integration with multiple index types (IVF_FLAT, HNSW, FLAT)
- ✅ **REST API**: FastAPI server with comprehensive endpoints
- ✅ **Type Safety**: Full type hints throughout the codebase
- ✅ **Production Ready**: Proper error handling, logging, and retry logic

## 📋 Performance Targets

| Stage             | Target                   |
| ----------------- | ------------------------ |
| JAX preprocess    | JIT-compiled, vectorized |
| Triton inference  | Batch optimized          |
| Insert throughput | ≥ 5k vectors/sec         |
| Search latency    | < 50 ms (TopK=10)        |

## 📦 Installation

### Prerequisites

- Python 3.8+
- Docker (for Triton and Milvus)
- CUDA-compatible GPU (optional but recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start Services

#### 1. Start Milvus

```bash
# Using Docker Compose (recommended)
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker-compose up -d

# Or using standalone container
docker run -d --name milvus \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:v2.3.0-latest \
  milvus run standalone
```

#### 2. Start Triton Inference Server

```bash
# Prepare your model repository
mkdir -p /tmp/triton_models/embedding_model/1

# Copy your ONNX model to the repository
# cp your_model.onnx /tmp/triton_models/embedding_model/1/model.onnx

# Copy Triton config
cp configs/triton_config.pbtxt /tmp/triton_models/embedding_model/config.pbtxt

# Start Triton server
docker run --gpus all -d --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /tmp/triton_models:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models
```

## 🎯 Quick Start

### Python API

```python
from src.pipeline import ImageEmbeddingPipeline
from src.config import ServiceConfig

# Load configuration
config = ServiceConfig.from_yaml('configs/config.yaml')

# Create pipeline
with ImageEmbeddingPipeline(config) as pipeline:
    
    # Create a collection
    pipeline.create_collection(
        name="my_images",
        dim=512,
        description="My image embeddings"
    )
    
    # Insert images
    image_paths = [
        "/path/to/image1.jpg",
        "/path/to/image2.jpg",
        "https://example.com/image3.jpg",  # URLs supported!
    ]
    
    ids = ["img_1", "img_2", "img_3"]
    metadata = [
        {"category": "nature", "source": "local"},
        {"category": "urban", "source": "local"},
        {"category": "portrait", "source": "web"},
    ]
    
    pipeline.insert_images(
        inputs=image_paths,
        ids=ids,
        metadata=metadata,
        collection_name="my_images",
    )
    
    # Search for similar images
    results = pipeline.search_images(
        query_input="/path/to/query.jpg",
        topk=5,
        collection_name="my_images",
    )
    
    for result in results:
        print(f"ID: {result['id']}, Score: {result['score']:.4f}")
```

### REST API

#### Start the API Server

```bash
# Using Python
python -m src.api_server

# Or with Uvicorn directly
uvicorn src.api_server:app --host 0.0.0.0 --port 8080
```

#### API Examples

**Health Check**
```bash
curl http://localhost:8080/health
```

**Create Collection**
```bash
curl -X POST http://localhost:8080/collections/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_images",
    "dim": 512,
    "description": "My image embeddings"
  }'
```

**Insert Images**
```bash
curl -X POST http://localhost:8080/insert \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      "/path/to/image1.jpg",
      "https://example.com/image2.jpg"
    ],
    "ids": ["img_1", "img_2"],
    "metadata": [
      {"category": "nature"},
      {"category": "urban"}
    ],
    "collection_name": "my_images"
  }'
```

**Search**
```bash
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{
    "query_input": "/path/to/query.jpg",
    "topk": 10,
    "collection_name": "my_images"
  }'
```

**Search with Filter**
```bash
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{
    "query_input": "/path/to/query.jpg",
    "topk": 10,
    "filter_expr": "metadata[\"category\"] == \"nature\"",
    "collection_name": "my_images"
  }'
```

### Command-Line Interface

Install the package:
```bash
pip install -e .
```

Use the CLI:

**Extract embeddings**
```bash
jax-embedding embed /path/to/image1.jpg /path/to/image2.jpg --output embeddings.npy
```

**Insert images**
```bash
jax-embedding insert /path/to/image*.jpg \
  --ids img_1 img_2 img_3 \
  --collection my_images \
  --create-collection
```

**Search for similar images**
```bash
jax-embedding search /path/to/query.jpg \
  --topk 10 \
  --collection my_images
```

**Manage collections**
```bash
# List collections
jax-embedding collection list

# Create collection
jax-embedding collection create --name my_images --dim 512

# Get stats
jax-embedding collection stats --name my_images

# Delete collection
jax-embedding collection delete --name my_images
```

**Health check**
```bash
jax-embedding health
```

## ⚡ GPU Acceleration

### Enable GPU for JAX Preprocessing

JAX preprocessing can leverage GPU for significant speedup (4-5x faster).

**Method 1: Via Configuration File**

Edit `configs/config.yaml`:
```yaml
preprocess:
  use_gpu: true          # Enable GPU acceleration
  jax_platform: "gpu"    # Explicitly specify GPU platform
```

**Method 2: Via Environment Variables**

```bash
export JAX_USE_GPU=true
export JAX_PLATFORM=gpu
```

**Method 3: Programmatic**

```python
from src.config import ServiceConfig, PreprocessConfig
from src.pipeline import ImageEmbeddingPipeline

# Configure GPU
config = ServiceConfig()
config.preprocess.use_gpu = True
config.preprocess.jax_platform = 'gpu'

# Pipeline will use GPU for preprocessing
pipeline = ImageEmbeddingPipeline(config)
```

### GPU Installation

For GPU support, install JAX with CUDA:

```bash
# For CUDA 11.x
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# For CUDA 12.x
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Verify GPU Usage

```python
import jax
print(jax.devices())  # Should show gpu devices

# Check in logs
# INFO: JAX configured to use GPU: gpu:0
```

**Note**: GPU acceleration works seamlessly with CPU fallback. If GPU is unavailable, preprocessing automatically uses CPU.

📖 **See [docs/JAX_GPU_ACCELERATION.md](docs/JAX_GPU_ACCELERATION.md) for complete guide**

## 📁 Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── preprocess_jax.py      # JAX preprocessing (jit + vmap)
│   ├── triton_client.py       # Triton Inference Server client
│   ├── milvus_client.py       # Milvus vector database client
│   ├── pipeline.py            # End-to-end orchestration
│   ├── api_server.py          # FastAPI REST endpoints
│   └── cli.py                 # Command-line interface
├── configs/
│   ├── config.yaml            # Service configuration
│   ├── triton_config.pbtxt    # Triton model config
│   └── milvus_config.yaml     # Milvus schema config
├── examples/
│   ├── basic_usage.py         # Basic usage examples
│   ├── batch_processing.py    # Batch processing benchmark
│   └── api_client.py          # API client examples
├── tests/
│   └── validate.py            # Validation tests
├── docs/
│   └── PERFORMANCE.md         # Performance optimization guide
├── docker-compose.yml         # Docker deployment
├── Dockerfile                 # API service container
├── setup.py                   # Package installation
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## ⚙️ Configuration

### Using YAML File

```yaml
triton:
  url: "localhost:8000"
  model_name: "embedding_model"
  protocol: "http"
  input_name: "input"    # Custom input tensor name
  output_name: "output"  # Custom output tensor name

milvus:
  host: "localhost"
  port: 19530
  embedding_dim: 512
  index_type: "IVF_FLAT"

preprocess:
  image_size: [224, 224]
  batch_size: 32
```

### Using Environment Variables

```bash
export TRITON_URL="localhost:8000"
export TRITON_MODEL_NAME="embedding_model"
export TRITON_INPUT_NAME="input"
export TRITON_OUTPUT_NAME="output"
export MILVUS_HOST="localhost"
export MILVUS_PORT="19530"
export MILVUS_EMBEDDING_DIM="512"
```

## 🔧 Advanced Usage

### Custom Triton Input/Output Node Names

If your Triton model uses custom input/output tensor names, you can specify them in the configuration:

```python
from src.config import ServiceConfig
from src.triton_client import TritonClient

# Method 1: Via configuration
config = ServiceConfig()
config.triton.input_name = "images"
config.triton.output_name = "embeddings"

# Method 2: Direct client instantiation
client = TritonClient(
    url="localhost:8000",
    model_name="my_model",
    input_name="custom_input",
    output_name="custom_output"
)

# The names are used as defaults, but can be overridden per call
embeddings = client.infer(
    inputs,
    input_name="different_input",  # Override for this call
    output_name="different_output"
)
```

### Custom Preprocessing

Custom preprocessors must inherit from `BaseJAXPreprocessor` for high-performance JAX-based preprocessing:

```python
from src import BaseJAXPreprocessor, ImageEmbeddingPipeline, ServiceConfig
import jax
import jax.numpy as jnp

class MyCustomPreprocessor(BaseJAXPreprocessor):
    """Your custom JAX-based preprocessor."""
    
    def __init__(self, image_size=(224, 224), **kwargs):
        super().__init__(image_size=image_size, **kwargs)
        # Your custom initialization
    
    def _preprocess_single_jax(self, image: jnp.ndarray) -> jnp.ndarray:
        """JAX-based preprocessing (automatically JIT-compiled and vectorized)."""
        # Use JAX operations for best performance
        resized = jax.image.resize(image, (*self.image_size, 3), method='bilinear')
        normalized = resized / 255.0
        return normalized

# Use your custom preprocessor with the pipeline
custom_preprocessor = MyCustomPreprocessor(
    image_size=(256, 256),
    use_gpu=True  # Enable GPU acceleration
)
config = ServiceConfig.from_yaml('configs/config.yaml')

pipeline = ImageEmbeddingPipeline(
    config=config,
    preprocessor=custom_preprocessor  # Must inherit from BaseJAXPreprocessor
)

# Use the pipeline normally
embeddings = pipeline.embed_images(image_paths)
```

For detailed guide and examples, see [Custom JAX Preprocessor Guide](docs/CUSTOM_JAX_PREPROCESSOR.md).

**Or use the built-in JAX preprocessor with custom parameters:**

```python
from src.preprocess_jax import JAXImagePreprocessor

# Custom preprocessing parameters
preprocessor = JAXImagePreprocessor(
    image_size=(299, 299),
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    cache_compiled=True,
)

# Preprocess single image
preprocessed = preprocessor.preprocess_single("/path/to/image.jpg")

# Batch preprocessing with vmap
batch_preprocessed = preprocessor.preprocess_batch([
    "/path/to/image1.jpg",
    "/path/to/image2.jpg",
    "https://example.com/image3.jpg",
])
```

### Async Inference

```python
from src.triton_client import TritonClient

client = TritonClient(
    url="localhost:8000",
    protocol="http",
)

# Async inference (HTTP only)
async_result = client.async_infer(
    inputs=preprocessed_batch,
    callback=lambda result: print(f"Done: {result}")
)
```

### Multiple Index Types

```python
# IVF_FLAT - Fast training, good recall
config.milvus.index_type = "IVF_FLAT"
config.milvus.nlist = 128
config.milvus.nprobe = 16

# HNSW - Best search quality
config.milvus.index_type = "HNSW"
config.milvus.M = 16
config.milvus.efConstruction = 256

# FLAT - Brute force (best for small datasets)
config.milvus.index_type = "FLAT"
```

## 📊 Performance Optimization

### JAX Preprocessing

- ✅ **JIT compilation**: Functions are compiled once and cached
- ✅ **Vectorization**: Batch processing with `vmap` for parallelization
- ✅ **No Python loops**: All operations are JAX-native
- ✅ **Pre-allocated buffers**: Minimized memory allocations

### Triton Inference

- ✅ **Dynamic batching**: Automatic request batching
- ✅ **GPU acceleration**: CUDA-optimized inference
- ✅ **Model optimization**: Support for TensorRT, ONNX Runtime
- ✅ **Retry logic**: Automatic retry on transient failures

### Milvus Operations

- ✅ **Batch insert**: High-throughput bulk insert
- ✅ **Auto flush**: Automatic data persistence
- ✅ **Index optimization**: Multiple index types for different use cases
- ✅ **Search optimization**: Configurable search parameters

## 🐛 Troubleshooting

### Triton Connection Issues

```python
# Check if Triton is running
curl http://localhost:8000/v2/health/ready

# Check model status
curl http://localhost:8000/v2/models/embedding_model/ready
```

### Milvus Connection Issues

```python
from pymilvus import connections, utility

# Test connection
connections.connect(host="localhost", port="19530")
print(utility.list_collections())
```

### JAX Device Issues

```python
import jax
print(jax.devices())  # Check available devices

# Force CPU if needed
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
```

## 📈 Benchmarking

Run the benchmark script to measure performance:

```bash
python examples/batch_processing.py
```

Expected output:
```
Embedding Extraction Performance:
  Total time: 2.150s
  Throughput: 46.5 images/sec
  Avg latency: 21.50ms per image

Insert Performance:
  Total time: 1.850s
  Throughput: 5405 vectors/sec

Search Performance (100 queries):
  Avg latency: 35.20ms
  P95 latency: 42.15ms
  P99 latency: 48.30ms
```

## 🔐 Security Considerations

- Never commit credentials or API keys
- Use environment variables for sensitive configuration
- Enable authentication for Milvus in production
- Use HTTPS for API server in production
- Implement rate limiting for public-facing APIs
- Validate and sanitize user inputs

## 🚀 Deployment

### Quick Start with Docker Compose

The easiest way to run the full stack:

```bash
# Start all services (Milvus, Triton, API)
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f embedding-api

# Stop all services
docker-compose down
```

### Manual Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/

CMD ["uvicorn", "src.api_server:app", "--host", "0.0.0.0", "--port", "8080"]
```

Build and run:

```bash
docker build -t image-embedding-service .
docker run -p 8080:8080 \
  -e TRITON_URL=triton:8000 \
  -e MILVUS_HOST=milvus \
  image-embedding-service
```

### Kubernetes Deployment

See `deployment/kubernetes/` for example manifests (to be added).

## 📚 API Documentation

Once the API server is running, visit:
- Interactive docs: http://localhost:8080/docs
- OpenAPI schema: http://localhost:8080/openapi.json

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📄 License

See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- JAX team for the amazing framework
- NVIDIA for Triton Inference Server
- Milvus team for the vector database
- FastAPI for the web framework

## 📞 Support

For issues and questions:
- GitHub Issues: [github.com/noahzhy/jaxEmbeddingMilvus/issues](https://github.com/noahzhy/jaxEmbeddingMilvus/issues)

---

Built with ❤️ for production ML systems