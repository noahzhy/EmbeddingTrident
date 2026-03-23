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
try:
    from turbojpeg import TurboJPEG
except Exception:
    TurboJPEG = None
from tenacity import Retrying, stop_after_attempt, wait_fixed, retry_if_exception

# --- 配置常量 ---
_BATCH_MAX_SIZE = int(os.getenv("IMAGE_PREPROCESS_MAX_BATCH_SIZE", "16"))
_BATCH_WAIT_TIMEOUT_S = float(
    os.getenv("IMAGE_PREPROCESS_BATCH_WAIT_TIMEOUT_S", "0.005"))
_HTTP_POOL_SIZE = int(os.getenv("IMAGE_PREPROCESS_HTTP_POOL_SIZE", "64"))
_IO_WORKERS = int(os.getenv("IMAGE_PREPROCESS_IO_WORKERS", "16"))
_USER_AGENT = "Mozilla/5.0"
_DOWNLOAD_TIMEOUT_S = 10
_DOWNLOAD_CHUNK_SIZE = 8192

# --- 全局工具对象 (延迟初始化) ---
_jpeg = None
_default_session = None


def _get_jpeg() -> TurboJPEG:
    global _jpeg
    if TurboJPEG is None:
        return None
    if _jpeg is None:
        _jpeg = TurboJPEG()
    return _jpeg


def _build_http_session(pool_size: int = _HTTP_POOL_SIZE) -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=pool_size,
        pool_maxsize=pool_size,
        max_retries=0,
        pool_block=False,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _get_default_http_session() -> requests.Session:
    global _default_session
    if _default_session is None:
        _default_session = _build_http_session(_HTTP_POOL_SIZE)
    return _default_session


def _should_retry_request(exc: BaseException) -> bool:
    if isinstance(exc, requests.exceptions.HTTPError):
        response = exc.response
        if response is None:
            return True
        status_code = response.status_code
        if status_code == 404:
            return False
        if status_code in (408, 429) or 500 <= status_code < 600: return True
        return False
    return isinstance(exc, requests.exceptions.RequestException)


def download(url: str, session: Optional[requests.Session] = None) -> bytes:
    http_session = session or _get_default_http_session()
    for attempt in Retrying(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry=retry_if_exception(_should_retry_request),
        reraise=True,
    ):
        with attempt:
            with http_session.get(
                url,
                timeout=_DOWNLOAD_TIMEOUT_S,
                stream=True,
                headers={"User-Agent": _USER_AGENT},
            ) as r:
                r.raise_for_status()
                return b"".join(chunk for chunk in r.iter_content(_DOWNLOAD_CHUNK_SIZE) if chunk)


def _read_bytes(src: str, session: Optional[requests.Session] = None) -> Union[bytes, np.ndarray]:
    if src.startswith("http://") or src.startswith("https://"):
        return download(src, session=session)
    if os.path.splitext(src)[1].lower() in [".jpg", ".jpeg"]:
        return np.fromfile(src, dtype=np.uint8)
    with open(src, "rb") as f:
        return f.read()


def _decode_image(src: str, data: Union[bytes, np.ndarray]) -> np.ndarray:
    ext = os.path.splitext(src)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        try:
            jpeg = _get_jpeg()
            if jpeg is not None:
                return jpeg.decode(data)
        except Exception:
            pass

    if isinstance(data, np.ndarray):
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    else:
        img = cv2.imdecode(np.frombuffer(
            memoryview(data), np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Failed to decode image: {src}")
    # force convert to RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fast_letterbox(img: np.ndarray, size=(224, 224), pad_value=114):
    h, w = img.shape[:2]
    th, tw = size
    r = min(th / h, tw / w)
    nh, nw = int(h * r), int(w * r)
    if (h, w) != (nh, nw):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    top = (th - nh) // 2
    bottom = th - nh - top
    left = (tw - nw) // 2
    right = tw - nw - left

    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value),
    )
    return img, {
        "input_shape": [int(th), int(tw)],
        "orig_shape": [int(h), int(w)],
        "pad_info": [float(left), float(top), float(r)],
    }


@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    max_ongoing_requests=100
)
class ImageLoaderNode:
    """Responsible for loading image from URL/Path and decoding. I/O intensive."""

    def __init__(self, io_workers: int = _IO_WORKERS):
        self.http_session = _build_http_session()
        self._io_pool = ThreadPoolExecutor(
            max_workers=io_workers, thread_name_prefix="loader-io")

    @batch(max_batch_size=_BATCH_MAX_SIZE, batch_wait_timeout_s=_BATCH_WAIT_TIMEOUT_S)
    async def handle_batch(self, urls: List[str]) -> List[np.ndarray]:
        loop = asyncio.get_running_loop()

        async def process_one(url):
            raw_bytes = await loop.run_in_executor(self._io_pool, _read_bytes, url, self.http_session)
            return _decode_image(url, raw_bytes)

        return await asyncio.gather(*(process_one(u) for u in urls))

    async def __call__(self, url: str) -> np.ndarray:
        return await self.handle_batch(url)


@serve.deployment(ray_actor_options={"num_cpus": 1})
class LetterboxNode:
    """Responsible for image resizing and padding. CPU intensive."""

    def __init__(self, target_size: tuple = (224, 224), pad_value: int = 114):
        self.target_size = target_size
        self.pad_value = pad_value

    async def __call__(self, img: np.ndarray) -> Any:
        return fast_letterbox(
            img,
            size=self.target_size,
            pad_value=self.pad_value,
        )


@serve.deployment(ray_actor_options={"num_cpus": 0.5})
class NormalizationNode:
    """Responsible for normalization and transpose."""

    async def __call__(self, img: np.ndarray) -> np.ndarray:
        # Optimized normalization logic as requested
        img = img.astype(np.float32) * (1/255.0)

        # Check if C is last dimension, if so transpose to CHW
        if img.shape[-1] == 3:
            img = img.transpose(2, 0, 1)

        return np.ascontiguousarray(img)

