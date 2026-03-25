import asyncio
import time
import os
import sys
from typing import Dict, List, Any

from starlette.requests import Request
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle


@serve.deployment(
    max_ongoing_requests=256,
    ray_actor_options={"num_cpus": 0.2},
)
class ModelInferPipeline:

    def __init__(
        self,
        preprocess: DeploymentHandle,
        infer: DeploymentHandle,
        pass_preprocess_result: bool = False,
    ):
        self.preprocess = preprocess
        self.infer = infer
        self.pass_preprocess_result = pass_preprocess_result

    @staticmethod
    def _extract_preprocess_image(preprocess_result: Any) -> Any:
        if isinstance(preprocess_result, dict):
            if "error" in preprocess_result and "image" not in preprocess_result:
                raise ValueError(str(preprocess_result["error"]))
            if "image" in preprocess_result:
                return preprocess_result["image"]
        return preprocess_result

    async def _infer_one_url(self, image_url: str) -> Any:
        preprocess_result = await self.preprocess.remote(image_url)
        if self.pass_preprocess_result:
            processed_image = preprocess_result
        else:
            processed_image = self._extract_preprocess_image(preprocess_result)
        return await self.infer.remote(processed_image)

    async def _run_urls_with_pipelining(self, image_urls: List[str]) -> List[Any]:
        if not image_urls:
            return []

        infer_responses = [self._infer_one_url(image_url) for image_url in image_urls]

        results = await asyncio.gather(*infer_responses, return_exceptions=True)
    
        clean_results = []
        for res in results:
            if isinstance(res, Exception):
                clean_results.append({"error": str(res)})
            else:
                clean_results.append(res)
                
        return clean_results

    @staticmethod
    def _strip_internal_buffers(payload: Any) -> Any:
        # Keep internal tensors for DAG chaining, but never expose them in HTTP JSON responses.
        def _clean(value: Any) -> Any:
            if isinstance(value, dict):
                cleaned: Dict[str, Any] = {}
                for key, item in value.items():
                    if key in {"raw_image", "image"}:
                        continue
                    cleaned[key] = _clean(item)
                return cleaned
            if isinstance(value, list):
                return [_clean(item) for item in value]
            return value

        return _clean(payload)

    async def __call__(self, request: Any) -> Dict:
        if isinstance(request, str):
            return await self._infer_one_url(request)

        if isinstance(request, list):
            return await self._run_urls_with_pipelining(request)

        try:
            data = await request.json()
        except Exception as e:
            return {"error": f"Invalid JSON payload: {str(e)}"}

        image_urls = data.get("image_urls")
        if image_urls is not None:
            if not isinstance(image_urls, list) or not image_urls:
                return {"error": "image_urls must be a non-empty list"}

            try:
                results = await self._run_urls_with_pipelining(image_urls)
                return {"results": self._strip_internal_buffers(results)}
            except Exception as exc:
                return {"error": str(exc)}

        image_url = data.get("image_url")
        if not image_url:
            return {"error": "Missing image_url or image_urls"}

        try:
            results = await self._run_urls_with_pipelining([image_url])
            return {"results": self._strip_internal_buffers(results)}
        except Exception as exc:
            return {"error": str(exc)}

