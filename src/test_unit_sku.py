import asyncio
import time
import os
import sys
import json
from urllib.request import urlopen
from urllib.error import URLError
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, List, Any

from starlette.requests import Request
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle
from src.nodes.image_node import *
from src.nodes.unit_node import UnitNode
from src.nodes.sku_node import SkuNode
from src.pipelines.image_pipeline import ImagePipeline
from src.pipelines.model_infer_pipeline import ModelInferPipeline
from utils.utils import wait_for_routes


if __name__ == "__main__":
    if not ray.is_initialized():
        try:
            ray.init(
                address="auto",
                _metrics_export_port=9090,
            )
        except Exception:
            ray.init()

    # Avoid stale detached Serve state (e.g., old host/port config) causing route drift.
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

    image_loader = ImageLoaderNode.bind(io_workers=16)
    letterbox_960 = LetterboxNode.bind(target_size=(960, 960))
    letterbox_224 = LetterboxNode.bind(target_size=(224, 224))
    normalization = NormalizationNode.bind()

    unit_infer = UnitNode.bind(
        triton_host="localhost",
        triton_port=8001,
        model_name="unit_ensemble",
    )
    sku_infer = SkuNode.bind(
        triton_host="localhost",
        triton_port=8001,
        model_name="Suntory-ES-Sku",
    )

    print("\n===== Unit inference pipeline =====")
    unit_preprocess = ImagePipeline.bind(
        image_loader,
        letterbox_960,
        normalization,
    )
    dag_unit = ModelInferPipeline.bind(
        unit_preprocess,
        unit_infer,
    )

    print("\n===== SKU inference pipeline =====")
    sku_preprocess = ImagePipeline.bind(
        image_loader,
        letterbox_224,
        normalization,
    )
    dag_sku = ModelInferPipeline.bind(
        sku_preprocess,
        sku_infer,
    )

    serve.run_many(
        [
            serve.RunTarget(
                target=dag_unit,
                name="dag_unit",
                route_prefix="/unit",
            ),
            serve.RunTarget(
                target=dag_sku,
                name="dag_sku",
                route_prefix="/sku",
            ),
        ]
    )

    wait_for_routes(
        base_url="http://127.0.0.1:2866",
        expected_routes=["/unit", "/sku"],
        timeout_s=45,
    )

    print("\n===== Dashboard running (Ctrl+C to exit) =====")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
