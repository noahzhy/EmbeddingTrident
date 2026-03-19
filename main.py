# dags_app.py

import asyncio
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle
from typing import Dict, Any

from src.nodes.image_node import ImagePreprocessNode
from src.nodes.sku_node import SkuNode
from src.nodes.unit_node import UnitNode
from src.pipelines.sku_infer_pipeline import SkuInferPipeline
from src.pipelines.unit_infer_pipeline import UnitInferPipeline


# ===============================
# =====   Router（统一入口）  =====
# ===============================

@serve.deployment(route_prefix="/infer")
class Router:
    def __init__(
        self,
        unit_pipeline: DeploymentHandle,
        sku_pipeline: DeploymentHandle,
    ):
        self.unit_pipeline = unit_pipeline
        self.sku_pipeline = sku_pipeline

    async def __call__(self, request):
        data = await request.json()

        scene = data.get("scene", "unit")

        # ===== DAG 路由 =====
        if scene == "unit":
            return await self.unit_pipeline.remote(data)

        elif scene == "sku":
            return await self.sku_pipeline.remote(data)

        # elif scene == "embedding":
        #     return await self.emb_pipeline.remote(data)

        else:
            return {"error": f"unknown scene: {scene}"}


# ===============================
# =====    DAG 构建（重点）   =====
# ===============================

def build_app():
    # ===== 共享 Node =====
    image_node = ImagePreprocessNode.bind()
    uni_node = UnitNode.bind(
        triton_host="localhost",
        triton_port=8001,
        model_name="unit_ensemble",
    )
    sku_node = SkuNode.bind(
        triton_host="localhost",
        triton_port=8001,
        model_name="Suntory-ES-Sku",
    )

    # ===== 多 DAG =====
    unit_pipeline = UnitInferPipeline.bind(
        image_node,
        uni_node,
    )

    sku_pipeline = SkuInferPipeline.bind(
        image_node,
        sku_node,
    )

    # ===== Router =====
    app = Router.bind(
        unit_pipeline,
        sku_pipeline,
    )

    return app


# ===============================
# ===== 启动入口 =====
# ===============================

if __name__ == "__main__":
    ray.init()
    serve.run(build_app())
