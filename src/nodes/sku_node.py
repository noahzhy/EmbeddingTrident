import os, sys
# add ../../
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import asyncio
from typing import Dict, List, Optional, Tuple, Union

import ray
import numpy as np
from ray import serve

try:
    from ray.serve.batch import batch
except Exception:
    from ray.serve import batch

from src.nodes.triton_node import TritonNode


TRITON_HOST = "localhost"
TRITON_PORT = 8001
SKU_MODEL_NAME = "Suntory-ES-Sku"


@serve.deployment(
    num_replicas=2,
    max_ongoing_requests=128,
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

            _, _, short_label = label.partition(":")
            decoded.append((short_label, score))

        return decoded

    async def _infer_single(self, image: np.ndarray) -> Dict:
        single_raw = await self.triton_client.infer_async(
            np.expand_dims(image, axis=0)
        )
        single_labels = self.post_process(single_raw)
        label = single_labels[0] if single_labels else ("", 0.0)
        return {"label": label[0], "score": label[1]}

    async def _infer_singles_concurrent(
        self, inputs: List[np.ndarray]
    ) -> List[Dict]:
        tasks = [self._infer_single(image) for image in inputs]
        return await asyncio.gather(*tasks)

    @batch(max_batch_size=128, batch_wait_timeout_s=0.01)
    async def infer_batch(
        self, inputs: List[np.ndarray]
    ) -> List[Dict]:
        """
        Expects a list of 3D [C, H, W] arrays, collected by the @batch decorator.
        Each call to __call__ enqueues exactly one 3D image here.
        """
        if not inputs:
            return []

        if any(not isinstance(image, np.ndarray) for image in inputs):
            raise TypeError("SkuNode expects numpy.ndarray inputs.")

        if any(image.ndim != 3 for image in inputs):
            raise ValueError("Each image must have shape [C, H, W].")

        if self._batch_supported is False:
            return await self._infer_singles_concurrent(inputs)

        input_array = np.stack(inputs)
        raw_results = await self.triton_client.infer_async(input_array)
        labels = self.post_process(raw_results)

        if len(labels) != len(inputs):
            self._batch_supported = False
            return await self._infer_singles_concurrent(inputs)

        self._batch_supported = True
        return [{"label": l[0], "score": l[1]} for l in labels]

    async def __call__(self, image: np.ndarray) -> Union[Dict, List[Dict]]:
        """
        Accepts either:
        - a single 3D image [C, H, W]  → returns a single Dict
        - a batch of 4D images [N, C, H, W] → returns a List[Dict]
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("SkuNode expects numpy.ndarray input.")

        # 4D input: split into individual 3D frames and fan out via infer_batch,
        # so each frame is enqueued as a separate item for the @batch decorator.
        if image.ndim == 4:
            tasks = [self.infer_batch(frame) for frame in image]
            return list(await asyncio.gather(*tasks))

        # 3D input: single image, enqueue directly
        if image.ndim == 3:
            return await self.infer_batch(image)

        raise ValueError(f"Unsupported input shape: {image.shape}. Expected [C,H,W] or [N,C,H,W].")


if __name__ == "__main__":
    ray.init()

    serve.start(
        http_options={
            "host": "127.0.0.1",
            "port": 0,
        },
    )

    print("\n===== SKU node testing =====")
    app = SkuNode.bind(
        triton_host=TRITON_HOST,
        triton_port=TRITON_PORT,
        model_name=SKU_MODEL_NAME,
    )
    handle = serve.run(app)

    # Test 1: list of individual 3D images
    images = [np.random.rand(3, 224, 224).astype(np.float32) for _ in range(32)]
    try:
        results = [handle.remote(img).result() for img in images]
        print("Test 1 - Inference result count:", len(results))
        print("Test 1 - Sample result:", results[0] if results else None)
    finally:
        pass

    import cv2
    # load all images from a directory and test batch inference
    im_dir = "data/debug_crops"
    images = [
        cv2.imread(os.path.join(im_dir, im_path))[:, :, ::-1]
        .transpose(2, 0, 1).astype(np.float32) / 255.0
        for im_path in sorted(os.listdir(im_dir))
        if im_path.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    im_np = np.stack(images)  # shape [N, C, H, W]
    try:
        results = handle.remote(im_np).result()
        print("Test 2 - Inference result count:", len(results))
        print("Test 2 - Sample result:", results[0] if results else None)
    finally:
        pass
