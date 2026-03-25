#!/bin/bash
# Send pipeline configuration to the control endpoint

ENDPOINT=${CONFIG_ENDPOINT:-http://localhost:2866/config}

echo "Sending pipeline config to $ENDPOINT ..."

curl -s -X POST "$ENDPOINT" \
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
    }' | python3 -m json.tool

echo ""
echo "Check status: curl -s $ENDPOINT/status | python3 -m json.tool"
