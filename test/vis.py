
import cv2
import json
import numpy as np
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    image_path = repo_root / "data" / "images" / "unit_test.jpg"
    json_path = repo_root / "data" / "unit_res.json"

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    H, W = image_bgr.shape[:2]

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    detections = data.get("detections", [])

    print(f"Loaded {len(detections)} detections from {json_path}")

    for det in detections:
        bbox = det["bbox"]
        score = det["score"]
        class_id = det["class_id"]
        # 反归一化
        x1 = int(bbox[0] * W)
        y1 = int(bbox[1] * H)
        x2 = int(bbox[2] * W)
        y2 = int(bbox[3] * H)
        color = (0, 255, 0)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{class_id}:{score:.3f}"
        cv2.putText(image_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # save the visualization result for reference
    cv2.imwrite(str(repo_root / "data" / "unit_test_detections.jpg"), image_bgr)


if __name__ == "__main__":
    main()
