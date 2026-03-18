import asyncio
import time
import os
import sys
from typing import Dict, List

import starlette
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.nodes.image_node import ImagePreprocessNode
from src.nodes.unit_node import UnitNode


@serve.deployment(
    max_ongoing_requests=256,
    ray_actor_options={"num_cpus": .2},
)
class UnitInferPipeline:

    def __init__(self, preprocess: DeploymentHandle, infer: DeploymentHandle):
        self.preprocess = preprocess
        self.infer = infer

    async def _run_urls_with_pipelining(self, image_urls: List[str]) -> List[Dict]:
        if not image_urls:
            return []

        # Pipeline style from base_pipeline.py: prefetch next preprocess task
        # while current item is being pushed to the infer stage.
        current_preprocess_ref = self.preprocess.remote(image_urls[0])
        infer_responses = []

        for image_url in image_urls[1:]:
            next_preprocess_ref = self.preprocess.remote(image_url)
            infer_responses.append(self.infer.remote(current_preprocess_ref))
            current_preprocess_ref = next_preprocess_ref

        infer_responses.append(self.infer.remote(current_preprocess_ref))
        return await asyncio.gather(*infer_responses)

    async def __call__(self, request: starlette.requests.Request) -> Dict:
        data = await request.json()

        image_urls = data.get("image_urls")
        if image_urls is not None:
            if not isinstance(image_urls, list) or not image_urls:
                return {"error": "image_urls must be a non-empty list"}

            try:
                results = await self._run_urls_with_pipelining(image_urls)
                return {"results": results}
            except Exception as exc:
                return {"error": str(exc)}

        image_url = data.get("image_url")
        if not image_url:
            return {"error": "Missing image_url"}

        try:
            # Keep intermediate tensor in Serve object refs to avoid extra hop/copy.
            processed_image_ref = self.preprocess.remote(image_url)
            infer_response = self.infer.remote(processed_image_ref)
            return await infer_response
        except Exception as exc:
            return {"error": str(exc)}


if __name__ == "__main__":
    ray.init()

    print("\n===== Unit inference pipeline testing =====")

    preprocess = serve.deployment(
        num_replicas=8,
        max_ongoing_requests=64,
        ray_actor_options={"num_cpus": 1},
    )(ImagePreprocessNode).bind(target_size=(960, 960), return_meta=True)
    unit_infer = UnitNode.bind(
        triton_host="localhost",
        triton_port=8001,
        model_name="unit_ensemble",
    )

    serve.start(
        detached=True,
        http_options={
            "host": "0.0.0.0",
            "port": 2867,
        },
    )

    # dag
    dag_app = UnitInferPipeline.bind(
        preprocess,
        unit_infer,
    )

    serve.run(dag_app, route_prefix="/")

    print("\n===== Dashboard running (Ctrl+C to exit) =====")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")