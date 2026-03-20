#!/bin/bash

rm -rf data/debug_crops
rm -rf data/unit_res.json
rm -rf data/unit_test_detections.jpg

# 1️⃣ 启动 pipeline（后台）
python src/main.py > pipeline.log 2>&1 &

PIPELINE_PID=$!
echo "Pipeline PID: $PIPELINE_PID"

# 2️⃣ 等待服务启动（很关键）
sleep 15

# 3️⃣ 发请求
bash scripts/req_unit_image.sh > data/unit_res.json

sleep 2

# 4️⃣ 可视化
python test/vis.py

# 5️⃣ 测试 crop
python src/nodes/crop_node.py

# 6️⃣ 可选：结束服务
kill $PIPELINE_PID