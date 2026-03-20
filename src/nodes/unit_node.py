import os, sys
## add ../ to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union

import ray
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
UNIT_MODEL_NAME = "CCTH-Unit"
UNIT_MODEL_TYPE = "yolov5"

UnitParamsSignature = Tuple[str, float, float, float, float, float, float, float]
UnitPayload = Tuple[np.ndarray, Dict[str, Any], UnitParamsSignature]


@serve.deployment(
    max_ongoing_requests=32,
    ray_actor_options={"num_cpus": 1},
)
class UnitNode:
    def __init__(
        self,
        triton_host: str = TRITON_HOST,
        triton_port: int = TRITON_PORT,
        model_name: str = UNIT_MODEL_NAME,
        model_type: str = UNIT_MODEL_TYPE,
        *args,
        **kwargs,
    ):
        self.triton_client = TritonNode(
            url=f"{triton_host}:{triton_port}",
            model_name=model_name,
            mode="unit",
        )
        self.model_type = model_type
        self._batch_supported: Optional[bool] = None

    @staticmethod
    def _build_params_signature(
        model_type: str,
        input_shape: List[float],
        orig_shape: List[float],
        pad_info: List[float],
    ) -> UnitParamsSignature:
        return (
            model_type,
            float(input_shape[0]),
            float(input_shape[1]),
            float(orig_shape[0]),
            float(orig_shape[1]),
            float(pad_info[0]),
            float(pad_info[1]),
            float(pad_info[2]),
        )

    @staticmethod
    def _to_float_list(value: Any, expected_len: int) -> Optional[List[float]]:
        if value is None:
            return None
        if isinstance(value, (list, tuple)) and len(value) >= expected_len:
            return [float(value[i]) for i in range(expected_len)]
        return None

    def _build_payload(
        self,
        item: Any,
    ) -> UnitPayload:
        if isinstance(item, dict):
            if "image" not in item:
                raise ValueError("UnitNode expects key 'image' when input payload is a dict")
            image = np.asarray(item.get("image"))
            input_shape = self._to_float_list(item.get("input_shape"), 2)
            orig_shape = self._to_float_list(item.get("orig_shape"), 2)
            pad_info = self._to_float_list(item.get("pad_info"), 3)
        else:
            image = np.asarray(item)
            input_shape = None
            orig_shape = None
            pad_info = None

        if image.ndim != 3:
            raise ValueError(
                f"Each image must have shape [C,H,W] or [H,W,C], got {image.shape}."
            )

        if input_shape is None and image.ndim >= 3:
            input_shape = [float(image.shape[-2]), float(image.shape[-1])]

        if input_shape is None:
            raise ValueError("Failed to infer input shape for unit inference request")

        if orig_shape is None:
            orig_shape = input_shape

        if pad_info is None:
            pad_info = [0.0, 0.0, 1.0]

        params: Dict[str, Any] = {
            "model_type": self.model_type,
            "input_shape": input_shape,
            "orig_shape": orig_shape,
            "pad_info": pad_info,
            "input_shape_h": input_shape[0],
            "input_shape_w": input_shape[1],
            "orig_shape_h": orig_shape[0],
            "orig_shape_w": orig_shape[1],
        }

        params_signature = self._build_params_signature(
            model_type=self.model_type,
            input_shape=input_shape,
            orig_shape=orig_shape,
            pad_info=pad_info,
        )

        return image, params, params_signature

    @staticmethod
    def post_process(raw_results: np.ndarray) -> List[List[Dict[str, Any]]]:
        if raw_results is None:
            return []

        arr = np.asarray(raw_results)
        if arr.size == 0:
            return []

        # Detection-only parsing: [x1, y1, x2, y2, score, class_id]
        if arr.ndim == 1:
            if arr.shape[0] < 6:
                return []
            arr = arr.reshape(1, 1, -1)
        elif arr.ndim == 2:
            if arr.shape[1] < 6:
                return []
            arr = arr.reshape(1, arr.shape[0], arr.shape[1])
        elif arr.ndim == 3:
            if arr.shape[2] < 6:
                return []
        else:
            return []

        batch_results: List[List[Dict[str, Any]]] = []
        for image_boxes in arr:
            detections: List[Dict[str, Any]] = []
            for row in image_boxes:
                score = float(row[4])
                if score <= 0:
                    continue

                detections.append(
                    {
                        "bbox": [
                            float(row[0]),
                            float(row[1]),
                            float(row[2]),
                            float(row[3]),
                        ],
                        "score": round(score, 4),
                        "class_id": int(row[5]),
                    }
                )

            batch_results.append(detections)

        return batch_results

    async def _infer_single_payload(
        self,
        payload: UnitPayload,
    ) -> Dict:
        image, params, _ = payload

        single_raw = await self.triton_client.infer_async(
            np.expand_dims(image, axis=0),
            params=params,
        )
        single_detections = self.post_process(single_raw)
        detections = single_detections[0] if single_detections else []
        return {
            "detections": detections,
        }

    async def _infer_payloads_concurrent(
        self,
        payloads: List[UnitPayload],
    ) -> List[Dict]:
        if not payloads:
            return []
        if len(payloads) == 1:
            return [await self._infer_single_payload(payloads[0])]

        tasks = [self._infer_single_payload(payload) for payload in payloads]
        return await asyncio.gather(*tasks)

    @batch(max_batch_size=8, batch_wait_timeout_s=0.01)
    async def infer_batch(self, inputs: List[Any]) -> List[Dict]:
        if not inputs:
            return []

        payloads = [self._build_payload(item) for item in inputs]

        if len(payloads) == 1:
            return [await self._infer_single_payload(payloads[0])]

        # If the model does not support true batch outputs, skip the wasted batch call.
        if self._batch_supported is False:
            return await self._infer_payloads_concurrent(payloads)

        first_signature = payloads[0][2]
        has_mixed_params = any(payload[2] != first_signature for payload in payloads[1:])
        if has_mixed_params:
            # A Triton request can only carry one params payload, so mixed runtime
            # params must be downgraded to per-sample inference for this batch.
            return await self._infer_payloads_concurrent(payloads)

        try:
            input_array = np.stack([payload[0] for payload in payloads])
        except ValueError:
            return await self._infer_payloads_concurrent(payloads)

        params = payloads[0][1]

        raw_results = await self.triton_client.infer_async(
            input_array,
            params=params,
        )
        detections_batch = self.post_process(raw_results) or []

        # Some models ignore batching and return detections for a single image only.
        if len(detections_batch) != len(inputs):
            self._batch_supported = False
            return await self._infer_payloads_concurrent(payloads)

        self._batch_supported = True

        return [{"detections": detections} for detections in detections_batch]

    async def __call__(self, image: Any) -> Union[Dict, List[Dict]]:
        if isinstance(image, np.ndarray):
            if image.ndim == 4:
                tasks = [self.infer_batch(frame) for frame in image]
                return list(await asyncio.gather(*tasks))

            if image.ndim == 3:
                return await self.infer_batch(image)

            raise ValueError(
                f"Unsupported input shape: {image.shape}. Expected [C,H,W] or [N,C,H,W]."
            )

        return await self.infer_batch(image)


if __name__ == "__main__":
    ray.init()

    serve.start(
        http_options={
            "host": "127.0.0.1",
            "port": 0,
        },
    )

    # test inference
    print("\n===== Unit inference node testing =====")
    app = UnitNode.bind(
        triton_host="localhost",
        triton_port=8001,
        model_name="unit_ensemble",
        # model_name="CCTH-Unit",
    )

    # bind() 返回的是 Application，不可直接调用；需 run() 后拿 handle
    handle = serve.run(app)

    from PIL import Image

    im_path = "data/images/unit_test.jpg"
    image = Image.open(im_path).convert("RGB")
    image = image.resize((960, 960))
    image_np = np.array(image).transpose(2, 0, 1).astype(np.float32)  # C,H,W, uint8 -> float32
    # normalize
    image_np = image_np / 255.0
    # double it as a batch of 2 to test batch inference
    # image_np = np.stack([image_np, image_np], axis=0)  #
    response = handle.remote(image_np)
    result = response.result() if hasattr(response, "result") else ray.get(response)
    print(f"Inference result: {result}")
