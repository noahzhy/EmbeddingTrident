# Quick Start Guide

Get the image embedding service running in 5 minutes.

## Prerequisites

- Python 3.8+
- Docker and Docker Compose
- NVIDIA GPU (optional but recommended)

## Option 1: Docker Compose (Recommended)

This will start Milvus, Triton, and the API service.

```bash
# Clone the repository
git clone https://github.com/noahzhy/jaxEmbeddingMilvus.git
cd jaxEmbeddingMilvus

# Start all services
docker-compose up -d

# Wait for services to be ready (30-60 seconds)
docker-compose ps

# Check API health
curl http://localhost:8080/health
```

Note: You'll need to add your model to `./triton_models/embedding_model/1/` before starting.

## Option 2: Local Development

### 1. Start Infrastructure

Start Milvus and Triton separately:

```bash
# Start Milvus
docker run -d --name milvus \
  -p 19530:19530 \
  milvusdb/milvus:v2.3.0-latest \
  milvus run standalone

# Start Triton (with your model)
docker run --gpus all -d --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /path/to/models:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models
```

### 2. Install Python Package

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install package
pip install -e .
```

### 3. Configure

```bash
# Copy example config
cp configs/config.yaml config.yaml

# Edit configuration
# Update Triton and Milvus URLs if needed
```

### 4. Run API Server

```bash
# Start the API server
uvicorn src.api_server:app --host 0.0.0.0 --port 8080
```

### 5. Verify

```bash
# Check health
curl http://localhost:8080/health

# API docs
open http://localhost:8080/docs
```

## Basic Usage

### Python API

```python
from src.pipeline import ImageEmbeddingPipeline
from src.config import ServiceConfig

# Initialize
config = ServiceConfig.from_yaml('config.yaml')
pipeline = ImageEmbeddingPipeline(config)

# Create collection
pipeline.create_collection("demo", dim=512)

# Insert images
pipeline.insert_images(
    inputs=["/path/to/img1.jpg", "/path/to/img2.jpg"],
    ids=["img1", "img2"],
    metadata=[{"type": "demo"}, {"type": "demo"}],
    collection_name="demo"
)

# Search
results = pipeline.search_images(
    query_input="/path/to/query.jpg",
    topk=5,
    collection_name="demo"
)

print(results)
```

### CLI

```bash
# Create collection
jax-embedding collection create --name demo --dim 512

# Insert images
jax-embedding insert img1.jpg img2.jpg \
  --ids img1 img2 \
  --collection demo

# Search
jax-embedding search query.jpg --topk 5 --collection demo
```

### REST API

```bash
# Create collection
curl -X POST http://localhost:8080/collections/create \
  -H "Content-Type: application/json" \
  -d '{"name": "demo", "dim": 512}'

# Insert images
curl -X POST http://localhost:8080/insert \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": ["/path/to/img1.jpg", "/path/to/img2.jpg"],
    "ids": ["img1", "img2"],
    "collection_name": "demo"
  }'

# Search
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{
    "query_input": "/path/to/query.jpg",
    "topk": 5,
    "collection_name": "demo"
  }'
```

## Troubleshooting

### Triton not connecting

```bash
# Check if Triton is running
curl http://localhost:8000/v2/health/ready

# Check model status
curl http://localhost:8000/v2/models/embedding_model/ready
```

### Milvus not connecting

```bash
# Check if Milvus is running
docker ps | grep milvus

# Test connection
python -c "from pymilvus import connections; connections.connect(host='localhost', port=19530); print('Connected!')"
```

### JAX/CUDA issues

```bash
# Check CUDA availability
python -c "import jax; print(jax.devices())"

# Force CPU if needed
export JAX_PLATFORMS=cpu
```

### Port conflicts

If ports 8080, 8000, or 19530 are in use:

```bash
# Option 1: Stop conflicting services
sudo lsof -i :8080
sudo kill -9 <PID>

# Option 2: Change ports in config
# Edit config.yaml or docker-compose.yml
```

## Next Steps

1. **Add your embedding model**: Place your ONNX model in `triton_models/`
2. **Tune configuration**: Edit `configs/config.yaml` for your use case
3. **Run benchmarks**: `python examples/batch_processing.py`
4. **Read performance guide**: See `docs/PERFORMANCE.md`
5. **Explore examples**: Check out `examples/` directory

## Common Commands

```bash
# View logs
docker-compose logs -f

# Restart service
docker-compose restart embedding-api

# Scale Triton instances
docker-compose up -d --scale triton=2

# Stop everything
docker-compose down

# Clean up volumes
docker-compose down -v
```

## Getting Help

- Documentation: See [README.md](../README.md)
- Performance tips: See [PERFORMANCE.md](PERFORMANCE.md)
- Issues: https://github.com/noahzhy/jaxEmbeddingMilvus/issues

---

Ready to scale? See the full [README](../README.md) for advanced deployment options.
