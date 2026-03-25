#!/bin/bash

set -euo pipefail

ENDPOINT="${ENDPOINT:-}"
OUT_PATH="data/vis_result.jpg"

echo "==================================="
echo "get raw unit_sku inference result for debugging"
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
        "http://0.0.0.0:2866/unit_sku"
        "http://127.0.0.1:2866/unit_sku"
    )
    if [[ -n "$HOST_IP" ]]; then
        CANDIDATES+=("http://${HOST_IP}:2866/unit_sku")
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

# print response body for debugging
echo "response body:"
cat "$TMP_JSON"
