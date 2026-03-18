# curl -X POST "http://127.0.0.1:2866/" \
# -H "Content-Type: application/json" \
# -d '{
#     "image_url": "https://f-pms-api.clobotics.cn/v/ed10bebc411f5edfcc5338f6a6a760f0.jpg"
# }'


# seq 1 64 | xargs -n1 -P4 -I{} curl -s -X POST "http://127.0.0.1:2866/" \
# -H "Content-Type: application/json" \
# -d '{"image_url":"https://f-pms-api.clobotics.cn/v/ed10bebc411f5edfcc5338f6a6a760f0.jpg"}'


# data/images/4653849.png

seq 1 128 | xargs -n1 -P2 -I{} curl -s -X POST "http://localhost:2866/" \
-H "Content-Type: application/json" \
-d '{"image_url":"data/images/4653849.png"}'

