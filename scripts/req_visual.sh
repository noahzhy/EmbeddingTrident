#!/bin/bash

set -euo pipefail

ENDPOINT="${ENDPOINT:-}"
OUT_PATH="data/vis_result.jpg"

echo "==================================="
echo "发送可视化请求并保存结果"
echo "==================================="

TMP_JSON=$(mktemp)
TMP_LAST=$(mktemp)

PAYLOAD='{
            "image_url": "data/images/unit_test.jpg",
            "app": "unit_sku"
        }'

if [[ -n "$ENDPOINT" ]]; then
    CANDIDATES=("$ENDPOINT")
else
    HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    CANDIDATES=(
        "http://0.0.0.0:2866/visual"
        "http://127.0.0.1:2866/visual"
    )
    if [[ -n "$HOST_IP" ]]; then
        CANDIDATES+=("http://${HOST_IP}:2866/visual")
    fi
fi

HTTP_CODE=""
USED_ENDPOINT=""
for candidate in "${CANDIDATES[@]}"; do
    code=$(curl -sS -o "$TMP_JSON" -w "%{http_code}" -X POST "$candidate" \
        -H "Content-Type: application/json" \
        -d "$PAYLOAD" || true)

    if [[ "$code" -ge 200 && "$code" -lt 300 ]]; then
        HTTP_CODE="$code"
        USED_ENDPOINT="$candidate"
        break
    fi

    HTTP_CODE="$code"
    cp "$TMP_JSON" "$TMP_LAST" || true
done

if [[ -z "$USED_ENDPOINT" && -s "$TMP_LAST" ]]; then
    cp "$TMP_LAST" "$TMP_JSON"
fi

if [[ -n "$USED_ENDPOINT" ]]; then
    echo "using endpoint: $USED_ENDPOINT"
fi

if [[ "$HTTP_CODE" -lt 200 || "$HTTP_CODE" -ge 300 ]]; then
    echo "request failed: http_status=${HTTP_CODE}" >&2
    echo "response body:" >&2
    cat "$TMP_JSON" >&2
    rm -f "$TMP_JSON" "$TMP_LAST"
    exit 1
fi

if [[ ! -s "$TMP_JSON" ]]; then
    echo "request failed: empty response body" >&2
    rm -f "$TMP_JSON" "$TMP_LAST"
    exit 1
fi

python - "$OUT_PATH" "$TMP_JSON" <<'PY'
import base64
import json
import sys

out_path = sys.argv[1]
json_path = sys.argv[2]
with open(json_path, "r", encoding="utf-8") as f:
    raw = f.read()

if not raw.strip():
    raise SystemExit("request failed: empty response body")

try:
    payload = json.loads(raw)
except json.JSONDecodeError:
    preview = raw[:400].replace("\n", "\\n")
    raise SystemExit(f"request failed: non-json response: {preview}")

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

rm -f "$TMP_JSON" "$TMP_LAST"

echo "请求完成！"
