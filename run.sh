source env.sh

# check RETAIL_MLMODELS_CONNECTION_STRING not None
if [ -z "$RETAIL_MLMODELS_CONNECTION_STRING" ]; then
    echo "Error: RETAIL_MLMODELS_CONNECTION_STRING is not set."
    exit 1
fi

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
