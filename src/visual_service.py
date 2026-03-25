import os
import time
from typing import List
from urllib.error import URLError
from urllib.request import urlopen
import json

import ray
from ray import serve

from src.nodes.visual_node import VisualNode
from src.pipelines.visual_pipeline import VisualPipeline
from utils.utils import wait_for_routes


def build_visual_app():
    visual_node = VisualNode.bind(top_k=5)
    visual_pipeline = VisualPipeline.bind(visual_node)
    return visual_pipeline


if __name__ == "__main__":
    if not ray.is_initialized():
        try:
            ray.init(address=os.getenv("RAY_ADDRESS", "auto"))
        except Exception:
            ray.init()

    if os.getenv("SERVE_CLEAN_START", "1") == "1":
        try:
            serve.shutdown()
            time.sleep(1)
        except Exception:
            pass

    serve.start(
        detached=True,
        http_options={
            "host": "0.0.0.0",
            "port": 2866,
        },
    )

    app = build_visual_app()
    serve.run(app, name="visual_app", route_prefix="/visualize")

    wait_for_routes(
        base_url="http://127.0.0.1:2866",
        expected_routes=["/visualize"],
        timeout_s=45,
    )

    print("Visual service is running on http://0.0.0.0:2866/visualize")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
