# Auto Model Evaluation Platform

## Project Overview

This project automates the evaluation of machine learning models (Object Detection & Classification) using **Ray Serve** for pipeline orchestration and **NVIDIA Triton Inference Server** for model serving. Models are downloaded from Azure Blob Storage, converted to Triton-compatible format, hot-loaded into Triton, and exposed as HTTP inference endpoints.

## Architecture

```
POST /config (JSON)
     │
     ▼
┌─────────────────┐     ┌──────────────┐     ┌────────────────┐
│ Config Endpoint │────▶│ Pipeline Mgr │────▶│ Triton Manager │
│  (Ray Serve)    │     │  download    │     │  HTTP load/    │
└─────────────────┘     │  gen config  │     │  unload API    │
                        │  deploy      │     └───────┬────────┘
                        └──────┬───────┘             │
                               │                     ▼
                               │              ┌─────────────┐
                               ▼              │   Triton     │
                        ┌─────────────┐       │   Server     │
                        │  Ray Serve  │──────▶│  (gRPC:8001) │
                        │  /unit      │       └─────────────┘
                        │  /unit_sku  │
                        │  /visual    │
                        └─────────────┘
```

## Quick Start (Auto-Config Mode)

One command to start everything:

```bash
source env.sh
bash run.sh
```

This will:
1. Start Triton in `explicit` model-control mode (empty repo)
2. Start Ray Serve with the `/config` control endpoint
3. Send a `POST /config` request to download models, generate Triton configs, hot-load, and deploy pipelines

### Customizing Models

Edit the JSON payload in `run.sh`, or send your own `POST /config` after startup:

```bash
# Start services first (without auto-POST)
bash scripts/start_triton.sh --explicit trt_models/
PIPELINE_AUTO_CONFIG=1 python3 src/main.py &

# Then configure with your models
curl -X POST http://localhost:2866/config \
  -H "Content-Type: application/json" \
  -d '{
    "models": [
      {
        "model_type": "Object_Detection",
        "model_name": "CCTH-Unit",
        "timestamp": "20260317083337"
      },
      {
        "model_type": "Classification",
        "model_name": "Suntory-ES-Sku",
        "timestamp": "20260318151232"
      }
    ]
  }'
```

Pipeline type is **auto-inferred** from the `model_type` combination:

| model_type 组合                   | Pipeline 类型 | 推理端点             |
| --------------------------------- | ------------- | -------------------- |
| Object_Detection only             | `unit`        | `/unit`              |
| Classification only               | `sku`         | `/sku`               |
| Object_Detection + Classification | `unit_sku`    | `/unit`, `/unit_sku` |

## Legacy Mode

To use the original manual workflow (hardcoded models, interactive Triton startup):

```bash
source env.sh
AUTO_CONFIG=0 bash run.sh
```

This runs the 4-step manual flow:
1. Download model files from blob storage
2. Generate Triton server configs (`generate_unit_triton.py` / `generate_sku_triton.py`)
3. Start Triton with interactive model selection
4. Start Ray Serve with hardcoded model names

## Control Plane API

All endpoints are served on port **2866**.

### Pipeline Configuration

| Method | Endpoint         | Description                            |
| ------ | ---------------- | -------------------------------------- |
| `POST` | `/config`        | Deploy pipeline from JSON model config |
| `GET`  | `/config`        | Return current active config           |
| `GET`  | `/config/status` | Triton + pipeline health status        |

### Inference Endpoints

| Method | Endpoint    | Description                         |
| ------ | ----------- | ----------------------------------- |
| `POST` | `/unit`     | Unit detection (Object_Detection)   |
| `POST` | `/unit_sku` | Unit detection + SKU classification |
| `POST` | `/visual`   | Visual search                       |

## Testing

### 1. Verify Services Are Running

```bash
# Check config endpoint
curl -s http://localhost:2866/config | python3 -m json.tool

# Check Triton health
curl -s http://localhost:8000/v2/health/ready

# Check pipeline + Triton status
curl -s http://localhost:2866/config/status | python3 -m json.tool
```

### 2. Configure Pipeline (POST /config)

```bash
# Use the convenience script
bash scripts/req_config.sh

# Or manually
curl -s -X POST http://localhost:2866/config \
  -H "Content-Type: application/json" \
  -d '{
    "models": [
      {"model_type": "Object_Detection", "model_name": "CCTH-Unit", "timestamp": "20260317083337"},
      {"model_type": "Classification", "model_name": "Suntory-ES-Sku", "timestamp": "20260318151232"}
    ]
  }' | python3 -m json.tool
```

Expected response:

```json
{
  "status": "deployed",
  "config": {
    "models": [...],
    "pipeline_type": "unit_sku",
    "unit_model_name": "CCTH-Unit",
    "sku_model_name": "Suntory-ES-Sku",
    "deployed_at": "2026-03-25T08:00:00+00:00"
  }
}
```

### 3. Run Inference

```bash
# Unit detection (single image)
curl -X POST http://localhost:2866/unit \
  -H "Content-Type: application/json" \
  -d '{"image_url": "data/images/unit_test.jpg"}'

# Unit detection (batch)
curl -X POST http://localhost:2866/unit \
  -H "Content-Type: application/json" \
  -d '{"image_urls": ["data/images/4653849.png", "data/images/unit_test.jpg"]}'

# Unit + SKU
curl -X POST http://localhost:2866/unit_sku \
  -H "Content-Type: application/json" \
  -d '{"image_url": "data/images/unit_test.jpg"}'
```

Or use the bundled scripts:

```bash
bash scripts/req_unit.sh
bash scripts/req_unit_sku.sh
bash scripts/req_sku.sh
```

### 4. Verify Persistence (Restart Recovery)

```bash
# After pipeline is deployed, restart Ray Serve
kill %1  # or Ctrl+C
python3 src/main.py &

# Pipeline auto-restores from config/active_pipeline.json
curl -s http://localhost:2866/config | python3 -m json.tool

# Inference should work without re-POST
curl -X POST http://localhost:2866/unit \
  -H "Content-Type: application/json" \
  -d '{"image_url": "data/images/unit_test.jpg"}'
```

### 5. Replace Pipeline with Different Models

```bash
curl -s -X POST http://localhost:2866/config \
  -H "Content-Type: application/json" \
  -d '{
    "models": [
      {"model_type": "Object_Detection", "model_name": "NewModel", "timestamp": "20260320160144"}
    ]
  }' | python3 -m json.tool

# Old models are unloaded, new pipeline is deployed
curl -s http://localhost:2866/config/status | python3 -m json.tool
```

## Project Structure

```
├── run.sh                          # Entry point (auto-config / legacy)
├── env.sh                          # Environment variables
├── config/
│   └── active_pipeline.json        # Persisted pipeline config (auto-generated)
├── src/
│   ├── main.py                     # Ray Serve application entry
│   ├── control_plane/
│   │   ├── config_endpoint.py      # POST/GET /config endpoint
│   │   ├── pipeline_manager.py     # Download → Triton config → deploy
│   │   └── triton_manager.py       # Triton HTTP API (load/unload)
│   ├── nodes/
│   │   ├── unit_node.py            # Object detection inference
│   │   ├── sku_node.py             # SKU classification inference
│   │   ├── triton_node.py          # Triton gRPC client
│   │   └── ...
│   └── pipelines/
│       ├── model_infer_pipeline.py # Preprocess → Infer pipeline
│       └── ...
├── scripts/
│   ├── start_triton.sh             # Start Triton (--explicit for hot-load)
│   ├── req_config.sh               # POST pipeline config
│   ├── req_unit.sh                 # Test unit endpoint
│   └── req_unit_sku.sh             # Test unit_sku endpoint
├── utils/
│   ├── blob_manager.py             # Azure Blob model downloader
│   ├── generate_unit_triton.py     # Generate Triton config (detection)
│   └── generate_sku_triton.py      # Generate Triton config (classification)
├── trt_models/                     # Generated Triton model repository
└── downloaded_models/              # Downloaded model artifacts
```

## Environment Variables

| Variable                 | Default                       | Description                               |
| ------------------------ | ----------------------------- | ----------------------------------------- |
| `PIPELINE_AUTO_CONFIG`   | `1`                           | `1` = auto-config mode, `0` = legacy mode |
| `AUTO_CONFIG`            | `1`                           | Same, used in `run.sh`                    |
| `TRITON_HTTP_URL`        | `http://localhost:8000`       | Triton HTTP API                           |
| `TRITON_GRPC_HOST`       | `localhost`                   | Triton gRPC host                          |
| `TRITON_GRPC_PORT`       | `8001`                        | Triton gRPC port                          |
| `TRITON_MODEL_REPO`      | `trt_models`                  | Triton model repository path              |
| `ACTIVE_MODELS_DIR`      | `.active_models`              | Container-mounted model dir               |
| `ACTIVE_PIPELINE_CONFIG` | `config/active_pipeline.json` | Persisted config path                     |
