source env.sh

# check RETAIL_MLMODELS_CONNECTION_STRING not None
if [ -z "$RETAIL_MLMODELS_CONNECTION_STRING" ]; then
    echo "Error: RETAIL_MLMODELS_CONNECTION_STRING is not set."
    exit 1
fi

# ================= Mode Selection =================
# Set AUTO_CONFIG=1 (default) for the new JSON POST workflow.
# Set AUTO_CONFIG=0 to use the legacy manual flow.
AUTO_CONFIG=${AUTO_CONFIG:-1}

if [ "$AUTO_CONFIG" = "1" ]; then
    echo "=== Auto-config mode ==="

    # Step 1: Start Triton in explicit model-control mode
    bash scripts/start_triton.sh --explicit trt_models/

    # Step 2: Start Ray Serve (with /config endpoint)
    export PIPELINE_AUTO_CONFIG=1
    python3 src/main.py &
    RAY_PID=$!

    # Step 3: Wait for Ray Serve to be ready
    echo "Waiting for Ray Serve..."
    for i in $(seq 1 30); do
        STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:2866/config 2>/dev/null)
        if [ "$STATUS" = "200" ]; then
            echo "Ray Serve is ready."
            break
        fi
        sleep 2
    done

    # Step 4: Send model configuration via POST
    echo "Sending pipeline configuration..."
    curl -s -X POST http://localhost:2866/config \
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

    wait $RAY_PID

else
    echo "=== Legacy mode ==="

    TARGET_FOLDER=downloaded_models

    MODEL_TYPE=Object_Detection
    MODEL_NAME=CCTH-Unit
    TIMESTAMP=20260317083337

    python3 utils/blob_manager.py \
        --model-type $MODEL_TYPE \
        --model-name $MODEL_NAME \
        --timestamp $TIMESTAMP \
        --target-folder $TARGET_FOLDER

    python utils/generate_unit_triton.py $TARGET_FOLDER/${MODEL_TYPE}/${MODEL_NAME}/${TIMESTAMP} trt_models \
        --model-name $MODEL_NAME \
        --onnx-model-name _$MODEL_NAME \
        --postprocess-model-name unit_postprocess

    MODEL_TYPE=Classification
    MODEL_NAME=Suntory-ES-Sku
    TIMESTAMP=20260318151232

    python3 utils/blob_manager.py \
        --model-type $MODEL_TYPE \
        --model-name $MODEL_NAME \
        --timestamp $TIMESTAMP \
        --target-folder $TARGET_FOLDER

    python utils/generate_sku_triton.py $TARGET_FOLDER/${MODEL_TYPE}/${MODEL_NAME}/${TIMESTAMP} trt_models \
        --model-name $MODEL_NAME

    bash scripts/start_triton.sh trt_models/

    export PIPELINE_AUTO_CONFIG=0
    python3 src/main.py
fi
