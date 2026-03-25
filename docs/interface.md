## API 文档（简版）

基于 [scripts/req_unit_sku.sh](scripts/req_unit_sku.sh) 和 [scripts/req_visual.sh](scripts/req_visual.sh) 整理。

## 1. 通用

- 协议：HTTP
- 方法：POST
- Header：`Content-Type: application/json`
- 默认服务地址：`http://10.198.199.142:2866`

请求体字段：

- `image_url`：图片路径或 HTTP 图片 URL
- `app`：应用标识，常用 `unit_sku`

示例（本地路径）：

```json
{
	"image_url": "data/images/unit_test.jpg",
	"app": "unit_sku"
}
```

示例（HTTP 图片）：

```json
{
	"image_url": "https://f-openapi.clobotics.cn//v/6cb8842b92e093263228a3be10ccf4e9.jpg",
	"app": "unit_sku"
}
```

## 2. `POST /unit_sku`

用途：获取原始识别结果（JSON）。

成功：状态码 `2xx`，返回 JSON。

响应 JSON 结构（当前实现）：

```json
{
	"results": [
		{
			"unit_results": [
				{
					"bbox": [0.328508, 0.695962, 0.39495, 0.823424],
					"score": 0.9768,
					"class_id": 0
				}
			],
			"sku_results": [
				{
					"label": "4657976",
					"score": 0.8095
				}
			],
			"unit_sku_results": [
				{
					"bbox": [0.328508, 0.695962, 0.39495, 0.823424],
					"score": 0.9768,
					"class_id": 0,
					"sku_label": "4657976",
					"sku_score": 0.8095
				}
			]
		}
	]
}
```

字段说明：

- `results`：`array`
- `results[0].unit_results`：单元检测结果，元素字段：`bbox/score/class_id`
- `results[0].sku_results`：SKU 分类结果，元素字段：`label/score`
- `results[0].unit_sku_results`：融合结果，元素字段：`bbox/score/class_id/sku_label/sku_score`

失败：

- 非 `2xx`：返回错误并打印响应体
- 空响应：`request failed: empty response body`

错误响应示例：

```json
{
  "error": "invalid image_url"
}
```

## 3. `POST /visual`

用途：获取可视化结果图。

成功：状态码 `2xx`，返回 JSON。

响应 JSON 结构（当前实现）：

```json
{
	"app": "unit_sku",
	"pipeline": "visual_pipeline",
	"pipeline_app": "unit_sku",
	"image_base64": "<base64-encoded-image>",
	"visual_source": "unit_sku_results",
	"result": {
		"results": [
			{
				"unit_results": [],
				"sku_results": [],
				"unit_sku_results": []
			}
		]
	}
}
```

字段说明：

- `app`：`string`，请求 app
- `pipeline`：`string`，处理 pipeline 名称
- `pipeline_app`：`string`，pipeline 对应 app
- `image_base64`：`string`，必填。base64 图片内容
- `visual_source`：`string`，本次实测值为 `unit_sku_results`
- `result`：`object`，内含推理结果（结构与 `/unit_sku` 相似）

脚本行为：`image_base64` 会被解码保存为 `data/vis_result.jpg`。

成功响应示例：

```json
{
	"app": "unit_sku",
	"pipeline": "visual_pipeline",
	"pipeline_app": "unit_sku",
	"image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
	"visual_source": "unit_sku_results",
	"result": {
		"results": [
			{
				"unit_results": [],
				"sku_results": [],
				"unit_sku_results": []
			}
		]
	}
}
```

失败：

- 非 `2xx`
- 空响应：`request failed: empty response body`
- 非 JSON：`request failed: non-json response: ...`
- 返回 `error` 字段：`request failed: <error>`
- 缺少 `image_base64`：`missing image_base64 in response`

错误响应示例：

```json
{
  "error": "model infer failed"
}
```

## 4. 响应结构示例

### 4.1 /unit_sku 成功（多目标）

```json
{
	"results": [
		{
			"unit_results": [
				{
					"bbox": [0.328508, 0.695962, 0.39495, 0.823424],
					"score": 0.9768,
					"class_id": 0
				}
			],
			"sku_results": [
				{
					"label": "4657976",
					"score": 0.8095
				}
			],
			"unit_sku_results": [
				{
					"bbox": [0.328508, 0.695962, 0.39495, 0.823424],
					"score": 0.9768,
					"class_id": 0,
					"sku_label": "4657976",
					"sku_score": 0.8095
				}
			]
		}
	]
}
```

### 4.2 /unit_sku 成功（空结果）

```json
{
	"results": [
		{
			"unit_results": [],
			"sku_results": [],
			"unit_sku_results": []
		}
	]
}
```

### 4.3 /visual 成功

```json
{
	"app": "unit_sku",
	"pipeline": "visual_pipeline",
	"pipeline_app": "unit_sku",
	"image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
	"visual_source": "unit_sku_results",
	"result": {
		"results": [
			{
				"unit_results": [],
				"sku_results": [],
				"unit_sku_results": []
			}
		]
	}
}
```

### 4.4 /visual 失败（缺少参数）

```json
{
	"error": "image_url is required"
}
```

### 4.5 /visual 失败（推理异常）

```json
{
	"error": "model infer failed"
}
```

## 5. 快速调用

```bash
# unit_sku
bash scripts/req_unit_sku.sh

# visual
bash scripts/req_visual.sh

# 覆盖 endpoint
ENDPOINT="http://10.198.199.142:2866/unit_sku" bash scripts/req_unit_sku.sh
ENDPOINT="http://10.198.199.142:2866/visual" bash scripts/req_visual.sh
```

HTTP 图片示例：

```bash
curl -X POST "http://10.198.199.142:2866/visual" \
	-H "Content-Type: application/json" \
	-d '{
		"image_url": "https://f-openapi.clobotics.cn//v/6cb8842b92e093263228a3be10ccf4e9.jpg",
		"app": "unit_sku"
	}'
```
