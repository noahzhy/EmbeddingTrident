# CV 模型推理与可视化服务平台

## 项目概述

本平台将 CV 模型的推理能力以 API 形式对外开放，服务两类核心用户：

- **数据团队**：通过标准化接口获取结构化推理结果，直接用于模型评估、数据分析与业务闭环，无需关心模型部署细节。
- **CV 团队**：通过可视化接口在实际图片上直观查看检测与分类效果，辅助模型调试与迭代。

## 核心能力

- **目标检测**：定位图像中的商品单元，返回 bounding box 及置信度
- **SKU 分类**：对检测区域进行品类识别，返回分类标签及置信度
- **联合推理**：单次请求同时执行检测与分类，支持单张及批量图片
- **可视化输出**：在原图上渲染检测框与标签，返回标注图像
- **多模型动态切换**：通过 API 在线更换模型版本，服务不中断

## API 调用示例

目标检测：
```http
POST /unit
{ "image_url": "https://example.com/shelf.jpg" }
→ [{ "bbox": [0.33, 0.70, 0.39, 0.82], "score": 0.98, "class_id": 0 }, ...]
```

检测 + 分类：
```http
POST /unit_sku
{ "image_url": "https://example.com/shelf.jpg" }
→ [{ "bbox": [...], "score": 0.98, "sku_label": "4657976", "sku_score": 0.46 }, ...]
```

可视化（含标注图像）：
```http
POST /visual
{ "image_url": "https://example.com/shelf.jpg", "app": "unit_sku" }
→ { "image_base64": "...", "results": [...] }
```

切换模型：
```http
POST /config
{ "models": [
    { "model_type": "Object_Detection", "model_name": "CCTH-Unit", "timestamp": "20260317" },
    { "model_type": "Classification",   "model_name": "Suntory-ES-Sku", "timestamp": "20260318" }
]}
```

## 技术特点

- 全流程通过 HTTP API 驱动，部署与推理均无需服务器端手动操作
- 支持模型在线热替换，切换过程中服务持续可用
- 部署配置自动持久化，服务异常恢复后无需重新配置
- 基于 NVIDIA Triton Inference Server 与 Ray Serve 构建，支持 GPU 推理加速与请求异步批处理
