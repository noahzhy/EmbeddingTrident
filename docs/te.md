整体流程如下：
```
                ┌────────────────────┐
image --------> │ YOLO ONNX (Triton) │
                └────────────────────┘
                           ↓
                ┌────────────────────┐
                │ Python Backend     │  ← decode + NMS + filter
                └────────────────────┘
                           ↓
                N x 6 boxes (normalized)
                           ↓
                crop → SKU classification
```


模型目录结构如下：
```
model_repository/
├── unit_ensemble/
│   ├── config.pbtxt
│   └── 1/
│
├── unit_onnx/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
│
├── unit_postprocess/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py
```