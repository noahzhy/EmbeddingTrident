#!/bin/bash

CONTAINER_NAME="triton_server"
IMAGE_NAME="nvcr.io/nvidia/tritonserver:25.04-py3"

# ================= Color Definitions =================
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # no color

# ================= Determine Model Repository =================
if [ -n "$1" ]; then
    ORIGINAL_MODEL_REPO="$1"
    echo -e "${BLUE}[INFO]${NC} Using provided model repository: $ORIGINAL_MODEL_REPO"
    if [ "$#" -gt 1 ]; then
        echo -e "${YELLOW}[WARN]${NC} Script takes only one argument (model repository path). Extra arguments ignored."
    fi
else
    ORIGINAL_MODEL_REPO="$(pwd)/models"
    echo -e "${BLUE}[INFO]${NC} Using default model repository: $ORIGINAL_MODEL_REPO"
fi

# Validate that the model repository directory exists
if [ ! -d "$ORIGINAL_MODEL_REPO" ]; then
    echo -e "${RED}[FAIL]${NC} Model repository directory not found at '$ORIGINAL_MODEL_REPO'.${NC}"
    exit 1
fi

ACTIVE_REPO="$(pwd)/.active_models"

# ================= Select Models =================
echo -e "${BLUE}[INFO]${NC} Checking available models in '$ORIGINAL_MODEL_REPO'...${NC}"

mapfile -t models < <(ls -1 "$ORIGINAL_MODEL_REPO")

if [ ${#models[@]} -eq 0 ]; then
    echo -e "${BLUE}[INFO]${NC} No models found in '$ORIGINAL_MODEL_REPO'. Nothing to start.${NC}"
    exit 0
fi

for i in "${!models[@]}"; do
    echo "  $((i+1))) ${models[$i]}"
done

echo -e "${BLUE}[INFO]${NC} Enter model numbers to start (space separated):${NC}"
read -r -p "> " selection
selected=()

for num in $selection; do
    if [[ "$num" =~ ^[0-9]+$ ]] && [ "$num" -ge 1 ] && [ "$num" -le "${#models[@]}" ]; then
        selected+=("${models[$((num-1))]}")
    else
        echo -e "${YELLOW}[WARN]${NC} Invalid input '$num' ignored.${NC}"
    fi
done

if [ ${#selected[@]} -eq 0 ]; then
    echo -e "${RED}[FAIL]${NC} No valid models selected, aborting.${NC}"
    exit 1
fi

# ================= Prepare Repo =================
echo -e "${BLUE}[INFO]${NC} Preparing model repository...${NC}"
rm -rf "$ACTIVE_REPO"
mkdir -p "$ACTIVE_REPO"

TOTAL_SIZE=0
for model in "${selected[@]}"; do
    MODEL_PATH="$ORIGINAL_MODEL_REPO/$model"
    if [ -d "$MODEL_PATH" ]; then
        cp -al "$MODEL_PATH" "$ACTIVE_REPO/" 2>/dev/null || cp -r "$MODEL_PATH" "$ACTIVE_REPO/"
        SIZE=$(du -sh "$MODEL_PATH" | awk '{print $1}')
        TOTAL_SIZE=$((TOTAL_SIZE + $(du -sb "$MODEL_PATH" | awk '{print $1}')))
        echo -e "${GREEN}[PASS]${NC} Copied $model | size: $SIZE${NC}"
    else
        echo -e "${RED}[FAIL]${NC} Model not found: $model${NC}"
    fi
done

NUM_MODELS=$(ls -1 "$ACTIVE_REPO" | wc -l)
echo -e "${BLUE}[INFO]${NC} Total models: $NUM_MODELS | Approx total size: $(numfmt --to=iec $TOTAL_SIZE)${NC}"

# ================= Start Container =================
echo -e "${BLUE}[INFO]${NC} Starting Triton Server...${NC}"
START_TIME=$(date +%s)

docker rm -f ${CONTAINER_NAME} 2>/dev/null

docker run -d \
    --name ${CONTAINER_NAME} \
    --gpus=all \
    --shm-size=1g \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v ${ACTIVE_REPO}:/models \
    ${IMAGE_NAME} \
    tritonserver \
    --model-repository=/models \
    --strict-model-config=false

# ================= Wait Ready =================
echo -e "${BLUE}[INFO]${NC} Waiting for Triton to be ready...${NC}"
sleep 2
for i in {1..15}; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v2/health/ready)
    if [ "$STATUS" = "200" ]; then
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        echo -e "${GREEN}[PASS]${NC} Triton is ready! | Startup time: ${ELAPSED}s${NC}"
        break
    fi
    echo -e "${BLUE}[INFO]${NC} Not ready yet (retry $i)...${NC}"
    sleep 2
done

# ================= Show Loaded Models =================
echo -e "${BLUE}[INFO]${NC} Loaded models:${NC}"
echo ""

JSON=$(curl -s -X POST http://localhost:8000/v2/repository/index)

if [[ -z "$JSON" || "$JSON" == "[]" ]]; then
    echo -e "${YELLOW}[WARN]${NC} No models loaded${NC}"
else
    printf "%-30s %-10s %-10s\n" "MODEL NAME" "VERSION" "STATE"
    echo "=========================================================="

    names=($(echo "$JSON" | grep -oP '"name":"\K[^"]+'))
    versions=($(echo "$JSON" | grep -oP '"version":"\K[^"]+'))
    states=($(echo "$JSON" | grep -oP '"state":"\K[^"]+'))

    for i in "${!names[@]}"; do
        state_color=$YELLOW
        if [[ "${states[$i]}" == "READY" ]]; then
            state_color=$GREEN
        elif [[ "${states[$i]}" == "UNAVAILABLE" ]]; then
            state_color=$RED
        fi
        printf "%-30s %-10s ${state_color}%-10s${NC}\n" \
            "${names[$i]}" "${versions[$i]}" "${states[$i]}"
    done
fi

echo ""
echo -e "${BLUE}[INFO]${NC} Logs: docker logs -f ${CONTAINER_NAME}${NC}"
