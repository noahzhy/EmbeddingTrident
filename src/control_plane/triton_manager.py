"""Triton Inference Server model management via HTTP API.

Requires Triton to be started with --model-control-mode=explicit.
"""

import time
from typing import Optional

import requests


class TritonModelManager:
    """Manages Triton model loading/unloading via the HTTP v2 API."""

    def __init__(self, triton_http_url: str = "http://localhost:8000"):
        self.base_url = triton_http_url.rstrip("/")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self, model_name: str) -> dict:
        """Load a model into Triton.

        POST /v2/repository/models/{model_name}/load
        """
        url = f"{self.base_url}/v2/repository/models/{model_name}/load"
        resp = requests.post(url, timeout=120)
        resp.raise_for_status()
        return resp.json() if resp.text else {}

    def unload_model(self, model_name: str) -> dict:
        """Unload a model from Triton.

        POST /v2/repository/models/{model_name}/unload
        """
        url = f"{self.base_url}/v2/repository/models/{model_name}/unload"
        resp = requests.post(url, timeout=30)
        resp.raise_for_status()
        return resp.json() if resp.text else {}

    def is_model_ready(self, model_name: str) -> bool:
        """Check whether a model is ready for inference.

        GET /v2/models/{model_name}/ready
        """
        url = f"{self.base_url}/v2/models/{model_name}/ready"
        try:
            resp = requests.get(url, timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def wait_model_ready(
        self,
        model_name: str,
        timeout: float = 120,
        interval: float = 2,
    ) -> bool:
        """Poll until a model is ready or *timeout* seconds have elapsed."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_model_ready(model_name):
                return True
            time.sleep(interval)
        return False

    def is_server_ready(self) -> bool:
        """Check whether the Triton server itself is healthy."""
        try:
            resp = requests.get(f"{self.base_url}/v2/health/ready", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list:
        """Return the repository index (all models known to Triton)."""
        url = f"{self.base_url}/v2/repository/index"
        resp = requests.post(url, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_loaded_model_names(self) -> list[str]:
        """Return names of models currently in READY state."""
        models = self.list_models()
        return [
            m["name"]
            for m in models
            if m.get("state") == "READY"
        ]
