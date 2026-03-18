import os
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import ray
import requests
import numpy as np
from ray import serve
from turbojpeg import TurboJPEG
from tenacity import Retrying, stop_after_attempt, wait_fixed, retry_if_exception_type


_jpeg = None


def _get_jpeg() -> TurboJPEG:
    global _jpeg
    if _jpeg is None:
        _jpeg = TurboJPEG()
    return _jpeg


def download(url: str) -> bytes:
    for attempt in Retrying(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        reraise=True,
    ):
        with attempt:
            print(f"Downloading: {url} (try {attempt.retry_state.attempt_number})")
            r = requests.get(
                url,
                timeout=10,
                stream=True,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            r.raise_for_status()

            data = bytearray()
            for chunk in r.iter_content(8192):
                if chunk:
                    data.extend(chunk)

            return bytes(data)


def _read_bytes(src: str):
    if src.startswith("http://") or src.startswith("https://"):
        return download(src)
    else:
        ext = os.path.splitext(src)[1].lower()
        if ext in [".jpg", ".jpeg"]:
            return np.fromfile(src, dtype=np.uint8)
        with open(src, "rb") as f:
            return f.read()


def load_image(src: str) -> np.ndarray:
    ext = os.path.splitext(src)[1].lower()
    if ext in [".jpg", ".jpeg"] and not (src.startswith("http://") or src.startswith("https://")):
        # np.fromfile + TurboJPEG is much faster than cv2.imread for local JPEG files
        data = _read_bytes(src)
        try:
            return _get_jpeg().decode(data)
        except Exception:
            pass

    data = _read_bytes(src)
    if isinstance(data, np.ndarray):
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    else:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Failed to decode image: {src}")

    return img


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

    def __init__(self, target_size: tuple = (224, 224), return_meta: bool = False):
        self.target_size = target_size
        self.return_meta = return_meta

    def __call__(self, url: str):
        img = load_image(url)
        if self.return_meta:
            img, meta = fast_letterbox(img, size=self.target_size, return_meta=True)
            img = transpose_and_normalize(img)
            payload: Dict[str, Any] = {
                "image": img,
                "input_shape": meta["input_shape"],
                "orig_shape": meta["orig_shape"],
                "pad_info": meta["pad_info"],
            }
            return payload

        img = fast_letterbox(img, size=self.target_size)
        img = transpose_and_normalize(img)
        return img


if __name__ == "__main__":
    if not ray.is_initialized():
        ray.init()

    print("\n===== Image preprocess node testing =====")
    node = ImagePreprocessNode()
    image_path = "data/images/4653849.png"
    result = node(image_path)
    # max
    max_val = np.max(result)
    print("Max pixel value after preprocessing:", max_val)
    print("Preprocessed image shape:", result.shape)
