import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import json
from typing import List
from urllib.error import URLError
from urllib.request import urlopen

import ray
from ray import serve

from src.nodes.image_node import *
from src.nodes.crop_node import CropNode
from src.nodes.unit_node import UnitNode
from src.nodes.sku_node import SkuNode
from src.nodes.visual_node import VisualNode
from src.pipelines.model_infer_pipeline import ModelInferPipeline
from src.pipelines.image_pipeline import ImagePipeline
from src.pipelines.visual_pipeline import VisualPipeline


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def wait_for_routes(base_url: str, expected_routes: List[str], timeout_s: int = 45, interval_s: float = 0.5) -> None:
    deadline = time.time() + timeout_s
    routes_url = f"{base_url}/-/routes"

    while time.time() < deadline:
        try:
            with urlopen(routes_url, timeout=2) as resp:
                route_payload = json.loads(resp.read().decode("utf-8"))
            
            available_routes = set(route_payload.keys()) if isinstance(
                route_payload, dict) else set()
            if all(route in available_routes for route in expected_routes):
                print(f"[ready] routes available: {sorted(available_routes)}")
                return
        except (URLError, TimeoutError, json.JSONDecodeError, OSError):
            pass
        
        time.sleep(interval_s)

    raise RuntimeError(
        f"Serve routes not ready within {timeout_s}s. Expected: {expected_routes}, url: {routes_url}"
    )


def build_unit_sku_pipeline():
    io_workers          = int(os.getenv("IMAGE_IO_WORKERS", 8))
    image_loader_cpus   = _env_float("IMAGE_LOADER_CPUS",   0.1)
    letterbox_cpus      = _env_float("LETTERBOX_CPUS",      0.5)
    normalization_cpus  = _env_float("NORMALIZATION_CPUS",  0.1)
    cropper_cpus        = _env_float("CROPPER_CPUS",        0.5)
    unit_cpus           = _env_float("UNIT_NODE_CPUS",      1.0)
    sku_cpus            = _env_float("SKU_NODE_CPUS",       1.0)
    image_pipeline_cpus = _env_float("IMAGE_PIPELINE_CPUS", 0.5)
    infer_pipeline_cpus = _env_float("INFER_PIPELINE_CPUS", 0.5)

    image_loader = ImageLoaderNode.options(
        ray_actor_options={"num_cpus": image_loader_cpus}
    ).bind(io_workers=io_workers)
    letterbox_960 = LetterboxNode.options(
        ray_actor_options={"num_cpus": letterbox_cpus}
    ).bind(target_size=(960, 960))
    cropper = CropNode.options(
        ray_actor_options={"num_cpus": cropper_cpus}
    ).bind(target_size=(224, 224))
    normalization = NormalizationNode.options(
        ray_actor_options={"num_cpus": normalization_cpus}
    ).bind()

    unit_infer = UnitNode.options(
        ray_actor_options={"num_cpus": unit_cpus}
    ).bind(
        triton_host="localhost",
        triton_port=8001,
        model_name="unit_ensemble",
    )
    sku_infer = SkuNode.options(
        ray_actor_options={"num_cpus": sku_cpus}
    ).bind(
        triton_host="localhost",
        triton_port=8001,
        model_name="Suntory-ES-Sku",
        cropper_node=cropper,
    )

    unit_preprocess = ImagePipeline.options(
        ray_actor_options={"num_cpus": image_pipeline_cpus}
    ).bind(
        image_loader,
        letterbox_960,
        normalization,
        include_raw_image=True,
    )

    unit_pipeline = ModelInferPipeline.options(
        name="unit_pipeline",
        ray_actor_options={"num_cpus": infer_pipeline_cpus},
    ).bind(
        preprocess=unit_preprocess,
        infer=unit_infer,
        pass_preprocess_result=True,
    )

    unit_sku_pipeline = ModelInferPipeline.options(
        name="unit_sku_pipeline",
        ray_actor_options={"num_cpus": infer_pipeline_cpus},
    ).bind(
        preprocess=unit_pipeline,
        infer=sku_infer,
        pass_preprocess_result=True,
    )

    return unit_sku_pipeline


def build_visual_pipeline():
    visual_node = VisualNode.bind(top_k=5)
    visual_pipeline = VisualPipeline.bind(visual_node)
    return visual_pipeline


class UnitSkuApplication:
    """
    Unit and SKU standard inference application using Ray Serve.
    """

    def __init__(self):
        if not ray.is_initialized():
            address = os.getenv("RAY_ADDRESS", "auto")
            min_cpu_required = _env_float("MIN_AVAILABLE_CPUS", 1.0)
            force_local_if_low_cpu = os.getenv(
                "RAY_FORCE_LOCAL_IF_LOW_CPU", "1") == "1"

            # 重启 ray
            print(f"[startup] initializing Ray with address='{address}'")
            ray.shutdown()
            time.sleep(1)

            try:
                ray.init(
                    address=address,
                    _metrics_export_port=9090,
                )
            except Exception:
                ray.init()

            available_cpu = float(ray.available_resources().get("CPU", 0.0))
            print(f"[startup] connected Ray available CPU: {available_cpu}")

            if force_local_if_low_cpu and available_cpu < min_cpu_required:
                print(
                    f"[startup] available CPU {available_cpu} < {min_cpu_required}, fallback to local Ray."
                )
                ray.shutdown()
                ray.init(num_cpus=max(2, (os.cpu_count() or 2)))

        if os.getenv("SERVE_CLEAN_START", "1") == "1":
            try:
                serve.shutdown()
                time.sleep(1)
                print("[startup] previous Serve instance cleaned")
            except Exception:
                pass

        serve.start(
            detached=True,
            http_options={
                "host": "0.0.0.0",
                "port": 2866,
            },
        )
        self.unit_sku_pipeline = build_unit_sku_pipeline()
        self.visual_pipeline = build_visual_pipeline()

    def start(self):
        print("Unit-SKU and visualization services are running on http://0.0.0.0:2866")
        serve.run_many(
            [
                serve.RunTarget(
                    target=self.unit_sku_pipeline,
                    name="unit_sku_app",
                    route_prefix="/infer_unit_sku",
                ),
                serve.RunTarget(
                    target=self.visual_pipeline,
                    name="visual_app",
                    route_prefix="/visualize",
                ),
            ]
        )
        wait_for_routes(
            base_url="http://127.0.0.1:2866",
            expected_routes=["/infer_unit_sku", "/visualize"],
            timeout_s=45,
        )
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Exiting...")


if __name__ == "__main__":
    app = UnitSkuApplication()
    app.start()
