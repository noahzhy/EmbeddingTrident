import os
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import ray
import requests
import numpy as np
from ray import serve

try:
    from ray.serve.batch import batch
except Exception:
    from ray.serve import batch

from ray.serve.exceptions import RayServeException
from turbojpeg import TurboJPEG
from tenacity import Retrying, stop_after_attempt, wait_fixed, retry_if_exception


_BATCH_MAX_SIZE = int(os.getenv("IMAGE_PREPROCESS_MAX_BATCH_SIZE", "16"))
_BATCH_WAIT_TIMEOUT_S = float(os.getenv("IMAGE_PREPROCESS_BATCH_WAIT_TIMEOUT_S", "0.005"))
_HTTP_POOL_SIZE = int(os.getenv("IMAGE_PREPROCESS_HTTP_POOL_SIZE", "64"))
_IO_WORKERS = int(os.getenv("IMAGE_PREPROCESS_IO_WORKERS", "16"))
_CPU_WORKERS = int(os.getenv("IMAGE_PREPROCESS_CPU_WORKERS", "4"))

_USER_AGENT = "Mozilla/5.0"
_DOWNLOAD_TIMEOUT_S = 10
_DOWNLOAD_CHUNK_SIZE = 8192


_jpeg = None
_default_session = None


RawImageBytes = Union[bytes, np.ndarray]


def _get_jpeg() -> TurboJPEG:
    global _jpeg
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

        if status_code in (408, 429) or 500 <= status_code < 600:
            return True

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
                return b"".join(
                    chunk for chunk in r.iter_content(_DOWNLOAD_CHUNK_SIZE) if chunk
                )


def _read_bytes(src: str, session: Optional[requests.Session] = None) -> RawImageBytes:
    if src.startswith("http://") or src.startswith("https://"):
        return download(src, session=session)

    ext = os.path.splitext(src)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        return np.fromfile(src, dtype=np.uint8)

    with open(src, "rb") as f:
        return f.read()


def _decode_image(src: str, data: RawImageBytes) -> np.ndarray:
    ext = os.path.splitext(src)[1].lower()

    if ext in [".jpg", ".jpeg"]:
        try:
            return _get_jpeg().decode(data)
        except Exception:
            pass


    if isinstance(data, np.ndarray):
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    else:
        img = cv2.imdecode(
            np.frombuffer(memoryview(data), np.uint8),
            cv2.IMREAD_COLOR,
        )

    if img is None:
        raise ValueError(f"Failed to decode image: {src}")

    return img


def load_image(src: str, session: Optional[requests.Session] = None) -> np.ndarray:
    data = _read_bytes(src, session=session)
    return _decode_image(src, data)


def _in_serve_replica_context() -> bool:
    try:
        serve.get_replica_context()
        return True
    except RayServeException:
        return False


def fast_letterbox(img: np.ndarray, size=(224, 224), pad_value=114, return_meta=False):
    h, w = img.shape[:2]
    th, tw = size
    r = min(th / h, tw / w)
    nh, nw = int(h * r), int(w * r)
    if (h, w) != (nh, nw):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    top    = (th - nh) // 2
    bottom = th - nh - top
    left   = (tw - nw) // 2
    right  = tw - nw - left

    img = cv2.copyMakeBorder(
        img,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )
    if return_meta:
        return img, {
            "input_shape": [int(th), int(tw)],
            "orig_shape": [int(h), int(w)],
            "pad_info": [float(left), float(top), float(r)],
        }
    return img


def transpose_and_normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32) * (1/255.0)
    return np.ascontiguousarray(img.transpose(2, 0, 1)) # HWC to CHW


class ImagePreprocessNode:

    def __init__(
        self,
        target_size: tuple = (224, 224),
        return_meta: bool = False,
        io_workers: int = _IO_WORKERS,
        cpu_workers: int = _CPU_WORKERS,
        http_pool_size: int = _HTTP_POOL_SIZE,
    ):
        self.target_size = target_size
        self.return_meta = return_meta
        self._closed = False
        self.http_session = _build_http_session(max(1, http_pool_size))

        self._io_pool = ThreadPoolExecutor(
            max_workers=max(1, io_workers),
            thread_name_prefix="image-io",
        )
        self._cpu_pool = ThreadPoolExecutor(
            max_workers=max(1, cpu_workers),
            thread_name_prefix="image-cpu",
        )

    def close(self):
        if self._closed:
            return
        self._closed = True

        session = getattr(self, "http_session", None)
        if session is not None:
            try:
                session.close()
            except Exception:
                pass

        for pool_name in ("_io_pool", "_cpu_pool"):
            pool = getattr(self, pool_name, None)
            if pool is None:
                continue
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                pool.shutdown(wait=False)
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _decode_and_preprocess_one(self, item: Tuple[str, RawImageBytes]) -> Any:
        src, raw_bytes = item
        img = _decode_image(src, raw_bytes)
        return self._preprocess_one(img)

    async def _pipeline_one(self, src: str) -> Any:
        if self._closed:
            raise RuntimeError("ImagePreprocessNode is already closed")

        loop = asyncio.get_running_loop()
        raw_bytes = await loop.run_in_executor(self._io_pool, _read_bytes, src, self.http_session)
        return await loop.run_in_executor(
            self._cpu_pool,
            self._decode_and_preprocess_one,
            (src, raw_bytes),
        )

    def _preprocess_one(self, img: np.ndarray) -> Any:
        if self.return_meta:
            img, meta = fast_letterbox(img, size=self.target_size, return_meta=True)
            img = transpose_and_normalize(img)
            payload: Dict[str, Any] = {
                "image":        img,
                "input_shape":  meta["input_shape"],
                "orig_shape":   meta["orig_shape"],
                "pad_info":     meta["pad_info"],
            }
            return payload

        img = fast_letterbox(img, size=self.target_size)
        img = transpose_and_normalize(img)
        return img

    @batch(max_batch_size=_BATCH_MAX_SIZE, batch_wait_timeout_s=_BATCH_WAIT_TIMEOUT_S)
    async def preprocess_batch(self, urls: List[str]) -> List[Any]:
        return await self._preprocess_batch_impl(urls)

    async def _preprocess_batch_impl(self, urls: List[str]) -> List[Any]:
        if not urls:
            return []

        tasks = [self._pipeline_one(url) for url in urls]
        return await asyncio.gather(*tasks)

    async def __call__(self, url: str):
        # @batch wrapper must run inside Serve replica context. Keep local script/test path usable.
        if not _in_serve_replica_context():
            outputs = await self._preprocess_batch_impl([url])
            return outputs[0]
        return await self.preprocess_batch(url)


if __name__ == "__main__":
    if not ray.is_initialized():
        ray.init()

    print("\n===== Image preprocess node testing =====")
    node = ImagePreprocessNode(target_size=(224, 224), return_meta=False)
    
    image_path = "data/images/4653849.png"
    try:
        result = asyncio.run(node(image_path))
        # max
        max_val = np.max(result)
        print("Max pixel value after preprocessing:", max_val)
        print("Preprocessed image shape:", result.shape)
    finally:
        node.close()
