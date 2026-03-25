"""Ray Serve control endpoint for dynamic pipeline configuration.

Routes:
  POST /config       — deploy new pipeline from JSON model config
  GET  /config       — return current active config
  GET  /config/status — return Triton + pipeline health
"""

import os
import sys
import traceback
from typing import Any, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from starlette.requests import Request
from starlette.responses import JSONResponse
from ray import serve

from src.control_plane.pipeline_manager import PipelineManager


@serve.deployment(
    max_ongoing_requests=4,
    ray_actor_options={"num_cpus": 0.1},
)
class ConfigEndpoint:
    """HTTP endpoint for pipeline configuration management."""

    def __init__(self):
        self._manager = PipelineManager()

    async def __call__(self, request: Request) -> JSONResponse:
        path = request.url.path.rstrip("/")

        if request.method == "GET":
            if path.endswith("/status"):
                return await self._get_status()
            return await self._get_config()

        if request.method == "POST":
            return await self._post_config(request)

        return JSONResponse({"error": "Method not allowed"}, status_code=405)

    async def _get_config(self) -> JSONResponse:
        status = self._manager.get_status()
        config = status.get("active_config")
        if config is None:
            return JSONResponse({"config": None, "message": "No active pipeline configured"})
        return JSONResponse({"config": config})

    async def _get_status(self) -> JSONResponse:
        status = self._manager.get_status()
        return JSONResponse(status)

    async def _post_config(self, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"error": "Invalid JSON payload"},
                status_code=400,
            )

        models = body.get("models")
        if not models or not isinstance(models, list):
            return JSONResponse(
                {"error": "'models' must be a non-empty list"},
                status_code=400,
            )

        # Validate each model entry
        required_keys = {"model_type", "model_name", "timestamp"}
        allowed_model_types = {"Object_Detection", "Classification"}
        for idx, m in enumerate(models):
            if not isinstance(m, dict):
                return JSONResponse(
                    {"error": f"models[{idx}] must be an object"},
                    status_code=400,
                )
            missing = required_keys - m.keys()
            if missing:
                return JSONResponse(
                    {"error": f"models[{idx}] missing keys: {missing}"},
                    status_code=400,
                )
            if m["model_type"] not in allowed_model_types:
                return JSONResponse(
                    {"error": f"models[{idx}].model_type must be one of {allowed_model_types}"},
                    status_code=400,
                )

        try:
            result = self._manager.build_and_deploy(models)
            return JSONResponse({
                "status": "deployed",
                "config": result,
            })
        except Exception as exc:
            traceback.print_exc()
            return JSONResponse(
                {"error": str(exc)},
                status_code=500,
            )
