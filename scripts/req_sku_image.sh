#!/bin/bash

ENDPOINT="http://0.0.0.0:2866/sku"

echo "==================================="
echo "发送单张图片请求 (image_url)"
echo "==================================="
curl -X POST "${ENDPOINT}" \
     -H "Content-Type: application/json" \
     -d '{
           "image_url": "data/debug_crops/crop_0001.jpg"
         }'

echo -e "\n\n==================================="
echo "发送多张图片批量请求 (image_urls)"
echo "==================================="
curl -X POST "${ENDPOINT}" \
     -H "Content-Type: application/json" \
     -d '{
           "image_urls": [
             "data/images/4653849.png",
             "data/debug_crops/crop_0001.jpg"
           ]
         }'
         
echo -e "\n\n请求完成！"
