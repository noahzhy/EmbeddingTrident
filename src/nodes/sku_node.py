import asyncio
from typing import Dict, List, Optional, Tuple

import numpy as np
from ray import serve

try:
    from ray.serve.batch import batch
except Exception:
    from ray.serve import batch

from src.nodes.triton_node import TritonNode


# TRITON_HOST = "10.2.250.89"
TRITON_HOST = "localhost"
TRITON_PORT = 8001
SKU_MODEL_NAME = "Suntory-ES-Sku"


@serve.deployment(
    num_replicas=2,
    max_ongoing_requests=32,
    ray_actor_options={"num_cpus": 1},
)
class SkuNode:
    def __init__(
        self,
        triton_host: str = TRITON_HOST,
        triton_port: int = TRITON_PORT,
        model_name: str = SKU_MODEL_NAME,
        *args,
        **kwargs,
    ):
        self.triton_client = TritonNode(
            url=f"{triton_host}:{triton_port}",
            model_name=model_name,
        )
        self._batch_supported: Optional[bool] = None

    @staticmethod
    def post_process(raw_results: np.ndarray) -> List[Tuple[str, float]]:
        arr = raw_results.reshape(-1)
        # vectorized decode
        arr = np.char.decode(arr.astype("S"), "utf-8")
        decoded: List[Tuple[str, float]] = []

        for value in arr:
            score_text, separator, label = value.partition(":")
            if not separator:
                decoded.append((value, 0.0))
                continue

            try:
                score = round(float(score_text), 4)
            except ValueError:
                score = 0.0

            decoded.append((label, score))

        return decoded

    async def _infer_single(self, image: np.ndarray) -> Dict:
        single_raw = await self.triton_client.infer_async(np.expand_dims(image, axis=0))
        single_labels = self.post_process(single_raw)
        label = single_labels[0] if single_labels else ("", 0.0)
        return {
            "label": label[0],
            "score": label[1],
        }

    async def _infer_singles_concurrent(self, inputs: List[np.ndarray]) -> List[Dict]:
        tasks = [self._infer_single(image) for image in inputs]
        return await asyncio.gather(*tasks)

    @batch(max_batch_size=8, batch_wait_timeout_s=0.01)
    async def infer_batch(self, inputs: List[np.ndarray]) -> List[Dict]:
        if not inputs:
            return []

        # If the model does not support true batch outputs, skip the wasted batch call.
        if self._batch_supported is False:
            return await self._infer_singles_concurrent(inputs)

        input_array = np.stack(inputs)
        raw_results = await self.triton_client.infer_async(input_array)
        labels = self.post_process(raw_results)

        # Some Triton models ignore batching and return a single output.
        if len(labels) != len(inputs):
            self._batch_supported = False
            return await self._infer_singles_concurrent(inputs)

        self._batch_supported = True

        return [{"label": l[0], "score": l[1]} for l in labels]

    async def __call__(self, image: np.ndarray) -> Dict:
        return await self.infer_batch(image)
