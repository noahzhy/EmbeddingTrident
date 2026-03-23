import base64
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from ray import serve


@serve.deployment(
    max_ongoing_requests=128,
    ray_actor_options={"num_cpus": 0.5},
)
class VisualNode:
    """Parse structured inference result and render visualization with OpenCV."""

    PRIORITY = [
        "unit_sku_results",
        "unit_results",
        "sku_results",
    ]

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    @staticmethod
    def _ensure_list(value: Any) -> List[Any]:
        return value if isinstance(value, list) else []

    def parse_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if isinstance(result, dict):
            results = result.get("results")
            if isinstance(results, list) and results:
                payload = results[0] if isinstance(results[0], dict) else {}
            else:
                payload = result

        parsed: Dict[str, Any] = {
            "source": None,
            "items": [],
        }

        for key in self.PRIORITY:
            items = self._ensure_list(payload.get(key))
            if items:
                parsed["source"] = key
                parsed["items"] = items
                break

        if parsed["source"] is None:
            parsed["source"] = "empty"

        return parsed

    @staticmethod
    def _to_pixel_bbox(bbox: List[float], width: int, height: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = [float(v) for v in bbox]

        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
            x1 *= width
            x2 *= width
            y1 *= height
            y2 *= height

        x1_i = int(np.clip(round(x1), 0, width - 1))
        y1_i = int(np.clip(round(y1), 0, height - 1))
        x2_i = int(np.clip(round(x2), 0, width - 1))
        y2_i = int(np.clip(round(y2), 0, height - 1))

        if x2_i <= x1_i:
            x2_i = min(width - 1, x1_i + 1)
        if y2_i <= y1_i:
            y2_i = min(height - 1, y1_i + 1)

        return x1_i, y1_i, x2_i, y2_i

    @staticmethod
    def _color_by_index(idx: int) -> Tuple[int, int, int]:
        palette = [
            (37, 99, 235),
            (5, 150, 105),
            (220, 38, 38),
            (234, 88, 12),
            (168, 85, 247),
            (14, 165, 233),
            (202, 138, 4),
            (236, 72, 153),
        ]
        return palette[idx % len(palette)]

    @staticmethod
    def _draw_label(image: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int]) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1

        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        y0 = max(0, y - th - baseline - 6)
        x0 = max(0, x)
        x1 = min(image.shape[1] - 1, x0 + tw + 6)
        y1 = min(image.shape[0] - 1, y0 + th + baseline + 6)

        cv2.rectangle(image, (x0, y0), (x1, y1), color, -1)
        cv2.putText(
            image,
            text,
            (x0 + 3, y1 - baseline - 3),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def _render_unit_sku(self, image: np.ndarray, items: List[Dict[str, Any]]) -> None:
        h, w = image.shape[:2]
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = self._to_pixel_bbox(list(bbox), w, h)
            color = self._color_by_index(idx)
            label = str(item.get("sku_label", "unknown"))
            score = float(item.get("sku_score", item.get("score", 0.0)))
            text = f"{label} {score:.3f}"

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            self._draw_label(image, text, x1, y1, color)

    def _render_unit(self, image: np.ndarray, items: List[Dict[str, Any]]) -> None:
        h, w = image.shape[:2]
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = self._to_pixel_bbox(list(bbox), w, h)
            color = self._color_by_index(idx)
            class_id = item.get("class_id", -1)
            score = float(item.get("score", 0.0))
            text = f"class:{class_id} {score:.3f}"

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            self._draw_label(image, text, x1, y1, color)

    def _render_sku_topk(self, image: np.ndarray, items: List[Dict[str, Any]]) -> None:
        lines: List[str] = []
        for item in items[: self.top_k]:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "unknown"))
            score = float(item.get("score", 0.0))
            lines.append(f"{label}: {score:.3f}")

        if not lines:
            lines = ["No SKU results"]

        x, y = 10, 24
        for i, line in enumerate(lines):
            color = self._color_by_index(i)
            self._draw_label(image, line, x, y + i * 24, color)

    def render(self, image: np.ndarray, parsed: Dict[str, Any]) -> np.ndarray:
        rendered = image.copy()
        source = parsed.get("source")
        items = parsed.get("items", [])

        if source == "unit_sku_results":
            self._render_unit_sku(rendered, items)
        elif source == "unit_results":
            self._render_unit(rendered, items)
        elif source == "sku_results":
            self._render_sku_topk(rendered, items)
        else:
            self._draw_label(rendered, "No results", 10, 24, (64, 64, 64))

        return rendered

    @staticmethod
    def _to_base64_jpeg(image: np.ndarray) -> str:
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("Failed to encode image to JPEG")
        return base64.b64encode(encoded.tobytes()).decode("utf-8")

    def __call__(self, image: np.ndarray, result: Dict[str, Any]) -> Dict[str, Any]:
        parsed = self.parse_result(result)
        rendered = self.render(image, parsed)
        return {
            "image_base64": self._to_base64_jpeg(rendered),
            "source": parsed.get("source"),
        }
