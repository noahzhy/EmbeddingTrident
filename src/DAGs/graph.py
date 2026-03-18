


if __name__ == "__main__":
    ray.init()

    serve.start(
        detached=True,
        http_options={
            "host": "0.0.0.0",
            "port": 2866,
        },
    )

    image_preprocess_node = serve.deployment(
        num_replicas=8,
        max_ongoing_requests=64,
        ray_actor_options={"num_cpus": 1},
    )(ImagePreprocessNode)

    dag_app = SkuInferPipeline.bind(
        image_preprocess_node.bind(),
        SkuInferNode.bind(),
    )

    serve.run(dag_app, route_prefix="/")

    print("\n===== SKU inference server running (Ctrl+C to exit) =====")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
