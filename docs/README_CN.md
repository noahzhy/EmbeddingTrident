# 使用说明

自动化 ML 模型评估平台。通过一个 JSON 请求完成：模型下载 → Triton 配置生成 → 热加载 → Ray Serve 推理部署。

## 快速启动

```bash
source env.sh
bash run.sh
```

启动后系统自动完成：启动 Triton（explicit 模式）→ 启动 Ray Serve → 部署默认 pipeline。

## 使用方式

所有接口端口 **2866**。

### 1. 部署 Pipeline

发送 POST 请求指定模型，系统自动完成下载、配置、加载、部署：

```bash
curl -s -X POST http://localhost:2866/config \
  -H "Content-Type: application/json" \
  -d '{
    "models": [
      {"model_type": "Object_Detection", "model_name": "CCTH-Unit", "timestamp": "20260317083337"},
      {"model_type": "Classification", "model_name": "Suntory-ES-Sku", "timestamp": "20260318151232"}
    ]
  }' | python3 -m json.tool
```

也可以用脚本：`bash scripts/req_config.sh`

Pipeline 类型根据 `model_type` 组合自动推断：

| model_type 组合                   | Pipeline   | 推理端点             |
| --------------------------------- | ---------- | -------------------- |
| Object_Detection only             | `unit`     | `/unit`              |
| Classification only               | `sku`      | `/sku`               |
| Object_Detection + Classification | `unit_sku` | `/unit`, `/unit_sku` |

### 2. 查看状态

```bash
# 当前配置
curl -s http://localhost:2866/config | python3 -m json.tool

# Triton + Pipeline 健康状态
curl -s http://localhost:2866/config/status | python3 -m json.tool
```

### 3. 推理请求

```bash
# 目标检测
curl -X POST http://localhost:2866/unit \
  -H "Content-Type: application/json" \
  -d '{"image_url": "data/images/unit_test.jpg"}'

# 目标检测 + SKU 分类
curl -X POST http://localhost:2866/unit_sku \
  -H "Content-Type: application/json" \
  -d '{"image_url": "data/images/unit_test.jpg"}'

# 批量推理
curl -X POST http://localhost:2866/unit \
  -H "Content-Type: application/json" \
  -d '{"image_urls": ["data/images/4653849.png", "data/images/unit_test.jpg"]}'
```

或使用脚本：`bash scripts/req_unit.sh` / `bash scripts/req_unit_sku.sh`

### 4. 切换模型

直接再发一次 POST /config，旧模型自动卸载，新模型自动加载：

```bash
curl -s -X POST http://localhost:2866/config \
  -H "Content-Type: application/json" \
  -d '{
    "models": [
      {"model_type": "Object_Detection", "model_name": "NewModel", "timestamp": "20260320160144"}
    ]
  }' | python3 -m json.tool
```

### 5. 重启恢复

Pipeline 配置自动持久化到 `config/active_pipeline.json`。重启 Ray Serve 后自动恢复，无需重新 POST。

## API 汇总

| 方法   | 端点             | 说明                        |
| ------ | ---------------- | --------------------------- |
| `POST` | `/config`        | 部署 pipeline（JSON 配置）  |
| `GET`  | `/config`        | 查看当前配置                |
| `GET`  | `/config/status` | 查看 Triton + Pipeline 状态 |
| `POST` | `/unit`          | 目标检测推理                |
| `POST` | `/unit_sku`      | 目标检测 + SKU 分类推理     |
| `POST` | `/visual`        | 视觉搜索                    |

## 环境变量

| 变量                | 默认值                  | 说明               |
| ------------------- | ----------------------- | ------------------ |
| `AUTO_CONFIG`       | `1`                     | `0` = 旧版手动模式 |
| `TRITON_HTTP_URL`   | `http://localhost:8000` | Triton HTTP 地址   |
| `TRITON_GRPC_HOST`  | `localhost`             | Triton gRPC 主机   |
| `TRITON_GRPC_PORT`  | `8001`                  | Triton gRPC 端口   |
| `TRITON_MODEL_REPO` | `trt_models`            | 模型仓库路径       |
