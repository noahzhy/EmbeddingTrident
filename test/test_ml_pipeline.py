import asyncio
import time
import os
import sys
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


if __name__ == "__main__":
    if not ray.is_initialized():
        try:
            ray.init(address="auto")
        except Exception:
            ray.init()


    serve.start(
        detached=True,
        http_options={
            "host": "0.0.0.0",
            "port": 2866,
        },
    )

    print("\n===== Unit inference pipeline testing =====")
    preprocess = ImagePipeline.bind(
        ImageLoaderNode.bind(io_workers=16),
        LetterboxNode.bind(target_size=(960, 960), return_meta=True),
        NormalizationNode.bind(),
    )
    unit_infer = UnitNode.bind(
        triton_host="localhost",
        triton_port=8001,
        model_name="unit_ensemble",
    )

    # dag
    dag_unit_app = ModelInferPipeline.bind(
        preprocess,
        unit_infer,
    )

    serve.run(dag_unit_app, route_prefix="/unit")


    print("\n===== SKU inference pipeline testing =====")
    preprocess = ImagePipeline.bind(
        ImageLoaderNode.bind(io_workers=16),
        LetterboxNode.bind(target_size=(224, 224)),
        NormalizationNode.bind(),
    )
    sku_infer = SkuNode.bind(
        triton_host="localhost",
        triton_port=8001,
        model_name="Suntory-ES-Sku",
    )
    dag_sku_app = ModelInferPipeline.bind(
        preprocess,
        sku_infer,
    )

    serve.run(dag_sku_app, route_prefix="/sku")


    print("\n===== Dashboard running (Ctrl+C to exit) =====")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")

