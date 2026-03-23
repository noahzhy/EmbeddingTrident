import asyncio
from typing import Any, Dict, Optional

import cv2
import numpy as np
import requests
from ray import serve
from ray.serve.handle import DeploymentHandle


@serve.deployment(
    max_ongoing_requests=128,
    ray_actor_options={"num_cpus": 0.5},
)
class VisualPipeline:
    """Ray Serve visualization pipeline for structured inference output."""

    def __init__(self, visual_node: DeploymentHandle, request_timeout_s: int = 10):
        self.visual_node = visual_node
        self.request_timeout_s = request_timeout_s

    @staticmethod
    def _decode_image_from_bytes(content: bytes) -> np.ndarray:
        arr = np.frombuffer(content, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode input image")
        return image

    def _load_image_sync(self, image_url: str) -> np.ndarray:
        if image_url.startswith("http://") or image_url.startswith("https://"):
            response = requests.get(image_url, timeout=self.request_timeout_s)
            response.raise_for_status()
            return self._decode_image_from_bytes(response.content)

        image = cv2.imread(image_url, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image from path: {image_url}")
        return image

    async def _load_image(self, image_url: str) -> np.ndarray:
        return await asyncio.to_thread(self._load_image_sync, image_url)

    @staticmethod
    async def _infer_by_deployment_handle(
        pipeline_name: str,
        image: np.ndarray,
        image_url: str,
        pipeline_app: str = "",
    ) -> Dict[str, Any]:
        def _needs_url_fallback(payload: Any) -> bool:
            if not isinstance(payload, dict):
                return False
            message = str(payload.get("error", ""))
            return "Invalid JSON payload" in message or "has no attribute 'json'" in message

        candidate_apps = []
        if pipeline_app:
            candidate_apps.append(pipeline_app)
        candidate_apps.extend(["unit_sku_app", "default", ""])

        last_error: Optional[Exception] = None
        for app_name in candidate_apps:
            try:
                if app_name:
                    handle = serve.get_deployment_handle(pipeline_name, app_name=app_name)
                else:
                    handle = serve.get_deployment_handle(pipeline_name)

                try:
                    # Preferred call style requested by the design doc.
                    result = await handle.remote(image)
                    if _needs_url_fallback(result):
                        return await handle.remote(image_url)
                    return result
                except Exception:
                    # Fallback for URL-based pipelines already deployed in the repo.
                    return await handle.remote(image_url)
            except Exception as exc:
                last_error = exc
                continue

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Failed to resolve deployment handle for pipeline={pipeline_name}")

    async def __call__(self, request: Any) -> Dict[str, Any]:
        if isinstance(request, dict):
            data = request
        else:
            data = await request.json()

        image_url = data.get("image_url")
        pipeline_name = data.get("pipeline")
        pipeline_app = str(data.get("pipeline_app", "")).strip()

        if not image_url:
            return {"error": "Missing image_url"}
        if not pipeline_name:
            return {"error": "Missing pipeline"}

        try:
            image = await self._load_image(image_url)
            result = await self._infer_by_deployment_handle(
                pipeline_name,
                image,
                image_url,
                pipeline_app=pipeline_app,
            )
            vis = await self.visual_node.remote(image, result)
            return {
                "pipeline": pipeline_name,
                "pipeline_app": pipeline_app,
                "image_base64": vis.get("image_base64", ""),
                "visual_source": vis.get("source", "empty"),
                "result": result,
            }
        except Exception as exc:
            return {
                "error": str(exc),
                "pipeline": pipeline_name,
                "image_url": image_url,
            }
