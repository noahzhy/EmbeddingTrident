import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import ray
import requests
import numpy as np
from ray import serve
from ray.serve.handle import DeploymentHandle

# 兼容旧版本 Ray 的 batch 导入
try:
    from ray.serve.batch import batch
except Exception:
    from ray.serve import batch

from ray.serve.exceptions import RayServeException
from turbojpeg import TurboJPEG
from tenacity import Retrying, stop_after_attempt, wait_fixed, retry_if_exception


@serve.deployment
class ImagePipeline:
    """Facade Node: Orchestrates the entire pipeline."""

    def __init__(
        self,
        loader: DeploymentHandle,
        letterbox: DeploymentHandle,
        normalizer: DeploymentHandle
    ):
        self.loader = loader
        self.letterbox = letterbox
        self.normalizer = normalizer

    async def __call__(self, url: str) -> Dict[str, Any]:
        # 1. Load image
        img = await self.loader.remote(url)

        # 2. Letterbox
        img_padded, meta = await self.letterbox.remote(img)

        # 3. Normalize
        final_tensor = await self.normalizer.remote(img_padded)

        return {
            "image": final_tensor,
            **meta
        }


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    from src.nodes.image_node import *

    if not ray.is_initialized():
        ray.init()

    # Start Serve with custom HTTP options
    serve.start(
        http_options={
            "host": "127.0.0.1",
            "port": 1234,
        },
    )

    # 1. Bind nodes
    loader_handle = ImageLoaderNode.bind(io_workers=16)
    letterbox_handle = LetterboxNode.bind(target_size=(224, 224))
    normalizer_handle = NormalizationNode.bind()

    # 2. Build pipeline
    pipeline = ImagePipeline.bind(
        loader_handle, letterbox_handle, normalizer_handle
    )

    # 3. Run Serve locally
    handle = serve.run(pipeline)

    print("\n===== 图像流水线测试中 =====")
    # Using local image path as requested
    test_url = "data/images/4653849.png"

    try:
        # Call the pipeline
        response = handle.remote(test_url)
        result = response.result()

        image_data = result["image"]
        # save to local for debugging
        debug_output_path = "debug_output.jpg"
        data = (image_data.transpose(1, 2, 0) * 255).astype(np.uint8)  # CHW to HWC and scale back to [0,255]
        cv2.imwrite(debug_output_path, data)
        print(f"处理成功!")
        print(f"输出形状: {image_data.shape} (预期: CHW)")
        print(f"像素最大值: {np.max(image_data):.4f}")
        print(f"元数据: { {k: v for k, v in result.items() if k != 'image'} }")
    except Exception as e:
        print(f"测试失败: {e}")
