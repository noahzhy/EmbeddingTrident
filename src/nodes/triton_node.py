import asyncio
import json
from typing import Any, Dict, List, Optional

import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from tritonclient.grpc import InferInput, InferRequestedOutput
from tritonclient.grpc import InferenceServerClient as SyncInferenceServerClient

try:
    from tritonclient.grpc.aio import InferenceServerClient as AsyncInferenceServerClient
except Exception:
    AsyncInferenceServerClient = None


class _ModelNotReadyError(RuntimeError):
    pass


class TritonNode:
    def __init__(
        self,
        url: str,
        model_name: str,
        input_name: Optional[str] = None,
        input_dtype: Optional[str] = None,
        output_name: Optional[str] = None,
        mode: str = "sku",
    ):
        self.url = url
        self.model_name = model_name
        self.input_name = input_name
        self.input_dtype = input_dtype
        self.output_name = output_name
        self.mode = mode

        self._client = SyncInferenceServerClient(url=self.url)
        self._async_client = AsyncInferenceServerClient(url=self.url) if AsyncInferenceServerClient else None

        # check model is exist and ready
        try:
            self._check_model()
        except _ModelNotReadyError as exc:
            raise ValueError(f"Model {self.model_name} is not ready: {exc}") from exc

        metadata = self._client.get_model_metadata(self.model_name)

        self.input_name = self.input_name or metadata.inputs[0].name
        self.input_dtype = self.input_dtype or metadata.inputs[0].datatype
        self.output_name = self.output_name or metadata.outputs[0].name
        self.outputs: Optional[List[InferRequestedOutput]] = None

        if self.mode == "sku":
            self.outputs = [InferRequestedOutput(self.output_name, class_count=1)]
        else:
            self.outputs = [InferRequestedOutput(self.output_name)]

        print(f"✅ Model: {self.model_name} | Input: {self.input_name} | Output: {self.output_name}")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(_ModelNotReadyError),
        reraise=True,
    )
    def _check_model(self) -> bool:
        try:
            ready = self._client.is_model_ready(self.model_name)
        except Exception as exc:
            raise _ModelNotReadyError(
                f"Failed to check readiness for model {self.model_name}: {exc}"
            ) from exc

        if ready:
            return True

        try:
            available_models = self._client.get_model_repository_index()
        except Exception as exc:
            available_models = f"<failed to list models: {exc}>"

        raise _ModelNotReadyError(
            f"Model {self.model_name} is not ready. Available models: {available_models}"
        )

    def _build_inputs(
        self,
        input_array: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[InferInput]:
        infer_input = InferInput(
            self.input_name,
            input_array.shape,
            self.input_dtype,
        )
        infer_input.set_data_from_numpy(input_array)
        inputs: List[InferInput] = [infer_input]

        if params is not None:
            payload = json.dumps(params, ensure_ascii=True)
            params_input = InferInput("params", [1], "BYTES")
            params_input.set_data_from_numpy(np.array([payload], dtype=object))
            inputs.append(params_input)

        return inputs

    def infer(
        self,
        input_array: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        inputs = self._build_inputs(input_array, params=params)
        res = self._client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=self.outputs,
        )
        return res.as_numpy(self.output_name)

    async def infer_async(
        self,
        input_array: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        if self._async_client is None:
            return await asyncio.to_thread(self.infer, input_array, params)

        inputs = self._build_inputs(input_array, params=params)
        res = await self._async_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=self.outputs,
        )
        return res.as_numpy(self.output_name)

    async def infer_safe(
        self,
        input_array: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
    ):
        try:
            return await self.infer_async(input_array, params=params)
        except Exception as exc:
            print(f"[WARN] async infer failed, fallback to sync thread: {exc}")
            return await asyncio.to_thread(self.infer, input_array, params)

    def close(self):
        if self._async_client is not None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._async_client.close())
            except RuntimeError:
                asyncio.run(self._async_client.close())
        self._client.close()

    async def close_async(self):
        if self._async_client is not None:
            await self._async_client.close()
        self._client.close()


if __name__ == "__main__":
    async def main():
        client = TritonNode(
            url="10.2.250.89:8001",
            model_name="Suntory-ES-Sku",
        )

        from PIL import Image

        image = Image.open("data/images/4653849.png").resize((224, 224))

        x = np.array(image).astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)

        result = await client.infer_async(x)

        value = result.reshape(-1)[0].decode("utf-8")
        label = value.split(":")[-1]

        print("Predicted:", label)

        await client.close_async()

    asyncio.run(main())