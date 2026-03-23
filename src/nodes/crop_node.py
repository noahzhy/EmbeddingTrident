# import torch
# import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Tuple

import ray
from ray import serve

import cv2
import json
from pathlib import Path

import sys
import os
# add ../..
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.nodes.image_node import fast_letterbox


@serve.deployment(ray_actor_options={"num_cpus": 1})
class CropNode:
    """
    专为 Ray DAG 优化的纯单线程 Numpy/OpenCV CropNode
    - 零多线程，完美契合 Ray 的底层进程调度
    - 向量化处理坐标反归一化与越界截断
    - 连续内存预分配，彻底消除 append 和 stack 的显存碎片与拷贝开销
    """
    def __init__(
        self,
        target_size=(224, 224),
        pad_value: int = 114,
    ):
        self.target_size = target_size
        self.pad_value = pad_value

    @staticmethod
    def _parse_detections_to_boxes(
        unit_results: Any,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if not isinstance(unit_results, list) or not unit_results:
            return np.empty((0, 6), dtype=np.float32), []

        boxes_rows: List[List[float]] = []
        kept_detections: List[Dict[str, Any]] = []
        for det in unit_results:
            if not isinstance(det, dict):
                continue
            bbox = det.get("bbox", [])
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue

            boxes_rows.append(
                [
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                    float(det.get("score", 0.0)),
                    float(det.get("class_id", -1)),
                ]
            )
            kept_detections.append(det)

        if not boxes_rows:
            return np.empty((0, 6), dtype=np.float32), []

        return np.asarray(boxes_rows, dtype=np.float32), kept_detections

    def _crop_core(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        H, W, C = image.shape
        N = boxes.shape[0]

        if N == 0:
            return np.empty((0, C, self.target_size[0], self.target_size[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)

        # 1. 向量化反归一化 (直接使用 numpy 底层 C 循环，极快)
        boxes_xyxy = boxes[:, :4].copy()
        if boxes_xyxy.max() <= 1.0:
            boxes_xyxy[:, [0, 2]] *= W
            boxes_xyxy[:, [1, 3]] *= H

        # 向量化越界截断 (in-place 原地修改，省内存)
        np.clip(boxes_xyxy[:, 0], 0, W, out=boxes_xyxy[:, 0])
        np.clip(boxes_xyxy[:, 1], 0, H, out=boxes_xyxy[:, 1])
        np.clip(boxes_xyxy[:, 2], 0, W, out=boxes_xyxy[:, 2])
        np.clip(boxes_xyxy[:, 3], 0, H, out=boxes_xyxy[:, 3])

        boxes_int = boxes_xyxy.astype(np.int32)

        # 2. 向量化过滤无效框 (避免在 Python 循环中做 if 判断)
        valid_mask = (boxes_int[:, 2] > boxes_int[:, 0]) & (boxes_int[:, 3] > boxes_int[:, 1])
        valid_indices = np.where(valid_mask)[0]
        valid_N = len(valid_indices)

        if valid_N == 0:
            return np.empty((0, C, self.target_size[0], self.target_size[1]), dtype=np.float32), valid_indices

        # 3. 【核心提速点】预先分配好整块连续的内存
        crops_array = np.empty((valid_N, C, self.target_size[0], self.target_size[1]), dtype=np.float32)

        # 4. 纯单线程紧凑循环
        for out_idx, idx in enumerate(valid_indices):
            x1, y1, x2, y2 = boxes_int[idx]
            crop = image[y1:y2, x1:x2]
            resized, _ = fast_letterbox(crop, size=self.target_size, pad_value=self.pad_value)
            # 转 CHW, 归一化，并直接按索引写入预分配的内存块中
            crops_array[out_idx] = resized.transpose(2, 0, 1).astype(np.float32) / 255.0

        return crops_array, valid_indices

    def __call__(self, *args, **kwargs) -> Any:
        # Compatible mode 1: CropNode(image, boxes) -> np.ndarray
        if len(args) == 2:
            image, boxes = args
            image_np = np.asarray(image)
            boxes_np = np.asarray(boxes, dtype=np.float32)
            crops, _ = self._crop_core(image_np, boxes_np)
            return crops

        # Compatible mode 2: CropNode({"raw_image": ..., "unit_results": [...]})
        # Returns dict with "image" to match ModelInferPipeline's extractor.
        if len(args) == 1 and isinstance(args[0], dict):
            payload = args[0]
            raw_image = payload.get("raw_image")
            if raw_image is None:
                raw_image = payload.get("image")

            image_np = np.asarray(raw_image)
            unit_results = payload.get("unit_results", [])
            boxes_np, kept_detections = self._parse_detections_to_boxes(unit_results)
            crops, valid_indices = self._crop_core(image_np, boxes_np)

            valid_detections = [kept_detections[i] for i in valid_indices.tolist()] if len(kept_detections) else []
            return {
                "image": crops,
                "unit_results": valid_detections,
            }

        raise TypeError("CropNode expects either (image, boxes) or a payload dict containing raw_image and unit_results")


# @serve.deployment()
# class CropDeployment:
#     """
#     Ray Serve Crop Deployment (去除了 PyTorch 依赖)
#     """
#     def __init__(self, target_size=(224, 224)):
#         self.node = CropNode(target_size=target_size)

#     async def __call__(self, request: dict):
#         """
#         request:
#             image: np.ndarray HWC uint8
#             boxes: np.ndarray (N,6) xyxy+score+class
#         """
#         img_np = request["image"]
#         boxes = request.get("boxes", np.empty((0, 6), dtype=np.float32))

#         # 直接传入 NumPy 数组处理
#         crops, valid_boxes = self.node(img_np, boxes)

#         return {
#             "crops": crops,
#             "boxes": valid_boxes
#         }


if __name__ == "__main__":
    node = CropNode(target_size=(224, 224))

    repo_root = Path(__file__).resolve().parents[2]
    image_path = repo_root / "data" / "images" / "unit_test.jpg"
    detection_json_path = repo_root / "data" / "unit_res.json"
    debug_dir = repo_root / "data" / "debug_crops"
    debug_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with detection_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    unit_results = payload.get("unit_results", [])
    boxes_list = []
    for det in unit_results:
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue
        boxes_list.append(
            [
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
                float(det.get("score", 0.0)),
                float(det.get("class_id", -1)),
            ]
        )

    boxes = np.asarray(boxes_list, dtype=np.float32)

    crops = node(image, boxes)

    for i, crop_chw in enumerate(crops):
        crop_hwc = np.transpose(crop_chw, (1, 2, 0))
        crop_u8 = np.clip(crop_hwc * 255.0, 0, 255).astype(np.uint8)
        crop_bgr = cv2.cvtColor(crop_u8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(debug_dir / f"crop_{i:04d}.jpg"), crop_bgr)

    print(f"Input image: {image_path}")
    print(f"Input unit_results: {len(unit_results)}")
    print("Crops shape:", crops.shape)
    print(f"Saved debug crops to: {debug_dir}")