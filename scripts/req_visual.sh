#!/bin/bash

set -euo pipefail

ENDPOINT="http://0.0.0.0:2866/visualize"
OUT_PATH="data/vis_result.jpg"

echo "==================================="
echo "发送可视化请求并保存结果"
echo "==================================="

RESP=$(curl -s -X POST "${ENDPOINT}" \
    -H "Content-Type: application/json" \
    -d '{
          "image_url": "data/images/unit_test.jpg",
          "pipeline": "unit_sku_pipeline",
          "pipeline_app": "unit_sku_app"
        }')

TMP_JSON=$(mktemp)
printf '%s' "$RESP" > "$TMP_JSON"

python - "$OUT_PATH" "$TMP_JSON" <<'PY'
import base64
import json
import sys

out_path = sys.argv[1]
json_path = sys.argv[2]
with open(json_path, "r", encoding="utf-8") as f:
    payload = json.load(f)

if "error" in payload:
    raise SystemExit(f"request failed: {payload['error']}")

image_b64 = payload.get("image_base64", "")
if not image_b64:
    raise SystemExit("missing image_base64 in response")

with open(out_path, "wb") as f:
    f.write(base64.b64decode(image_b64))

print(f"saved visualization to: {out_path}")
print(f"visual_source: {payload.get('visual_source')}")
PY

rm -f "$TMP_JSON"

echo "请求完成！"
