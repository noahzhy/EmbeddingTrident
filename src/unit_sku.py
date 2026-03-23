import time
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ray
from ray import serve
from src.nodes.image_node import *
from src.nodes.unit_node import UnitNode
from src.nodes.sku_node import SkuNode
from src.nodes.crop_node import CropNode
from src.pipelines.image_pipeline import ImagePipeline
from src.pipelines.model_infer_pipeline import ModelInferPipeline


class UnitSkuApplication:
    """
    Unit and SKU standard inference application using Ray Serve.
    """
    def __init__(self):
        if not ray.is_initialized():
            try:
                ray.init(
                    address="auto",
                    _metrics_export_port=9090,
                )
            except Exception:
                ray.init()

        serve.start(
            detached=True,
            http_options={
                "host": "0.0.0.0",
                "port": 2866,
            },
        )
        self.unit_sku_pipeline = build_unit_sku_pipeline()

    def start(self):
        print("Unit-SKU inference application is running on http://0.0.0.0:2866")
        serve.run(self.unit_sku_pipeline, route_prefix="/infer_unit_sku")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Exiting...")


image_loader = ImageLoaderNode.bind(io_workers=16)
letterbox_960 = LetterboxNode.bind(target_size=(960, 960))
cropper = CropNode.bind(target_size=(224, 224))
normalization = NormalizationNode.bind()


def build_unit_sku_pipeline():
    unit_infer = UnitNode.bind(
        triton_host="localhost",
        triton_port=8001,
        model_name="unit_ensemble",
    )
    sku_infer = SkuNode.bind(
        triton_host="localhost",
        triton_port=8001,
        model_name="Suntory-ES-Sku",
        cropper_node=cropper,
    )

    unit_preprocess = ImagePipeline.bind(
        image_loader,
        letterbox_960,
        normalization,
        include_raw_image=True,
    )

    print("\n===== Unit inference pipeline =====")
    print("\n===== SKU inference pipeline =====")
    unit_pipeline = ModelInferPipeline.options(name="unit_pipeline").bind(
        preprocess=unit_preprocess,
        infer=unit_infer,
        pass_preprocess_result=True,
    )

    unit_sku_pipeline = ModelInferPipeline.options(name="unit_sku_pipeline").bind(
        preprocess=unit_pipeline,
        infer=sku_infer,
        pass_preprocess_result=True,
    )

    return unit_sku_pipeline


if __name__ == "__main__":
    app = UnitSkuApplication()
    app.start()
