import os
import time

import jax.numpy as jnp
import numpy as np
import yaml
import ray
from loguru import logger
from ray.data import ActorPoolStrategy

from preprocess import build_preprocess_batch, extract_image_size
from milvus_ops import MilvusInserter, build_index_and_load, create_collection_no_index
from triton_ops import TritonClient


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    def first(items, default=None):
        return items[0] if items else default

    pipelines = raw.get("pipelines", {})
    triton = raw.get("triton_server", {})
    milvus = raw.get("milvus_server", {})
    model = first(raw.get("models", []), {})
    input_cfg = first(model.get("inputs", []), {})
    output_cfg = first(model.get("outputs", []), {})
    collection = first(raw.get("collections", []), {})
    fields = collection.get("fields", [])

    primary_field = next((f for f in fields if f.get("primary_key")), {})
    vector_field = next((f for f in fields if f.get("type") == "FLOAT_VECTOR"), {})
    index_field = next((f for f in fields if f.get("index")), {})

    output_shape = output_cfg.get("shape") or []
    vector_dim = vector_field.get("dim")
    if vector_dim is None and output_shape:
        vector_dim = output_shape[-1]

    return {
        "JAX_MEM_FRACTION": str(pipelines.get("jax_mem_fraction", 0.2)),
        "BATCH_SIZE": int(pipelines.get("batch_size", 128)),
        "NUM_GPUS_PER_WORKER": float(pipelines.get("num_gpus_per_worker", 0.5)),
        "WORKER_POOL_SIZE": tuple(pipelines.get("worker_pool_size", [1, 2])),
        "TRITON_URL": f"{triton.get('host', 'localhost')}:{triton.get('port', 8001)}",
        "MILVUS_HOST": milvus.get("host", "localhost"),
        "MILVUS_PORT": int(milvus.get("port", 19530)),
        "MODEL_NAME": model.get("name", "emb_siglip2"),
        "INPUT_NAME": input_cfg.get("name", "pixel_values"),
        "INPUT_DTYPE": input_cfg.get("dtype", "FP32"),
        "INPUT_SHAPE": input_cfg.get("shape") or [-1, 3, 224, 224],
        "OUTPUT_NAME": output_cfg.get("name", "image_embeds"),
        "OUTPUT_DTYPE": output_cfg.get("dtype", "FP32"),
        "COLLECTION_NAME": collection.get("name", "test_collection"),
        "COLLECTION_FIELDS": fields,
        "COLLECTION_AUTO_ID": bool(primary_field.get("auto_id", False)),
        "COLLECTION_VECTOR_DIM": int(vector_dim or 768),
        "COLLECTION_INDEX_FIELD": index_field.get("name"),
        "COLLECTION_INDEX_CFG": index_field.get("index"),
    }


class Processor:
    def __init__(self, config):
        self.config = config
        self.image_size = extract_image_size(config["INPUT_SHAPE"])
        self.preprocess_batch = build_preprocess_batch(self.image_size)

        self.triton = TritonClient(
            url=self.config["TRITON_URL"],
            model_name=self.config["MODEL_NAME"],
            input_name=self.config["INPUT_NAME"],
            input_dtype=self.config["INPUT_DTYPE"],
            output_name=self.config["OUTPUT_NAME"],
        )

        self.milvus = MilvusInserter(
            host=self.config["MILVUS_HOST"],
            port=self.config["MILVUS_PORT"],
            collection_name=self.config["COLLECTION_NAME"],
            auto_id=self.config["COLLECTION_AUTO_ID"],
        )

        logger.debug("[Worker {}] Warming up JAX JIT and Triton connection...", os.getpid())
        dummy_data = jnp.zeros((1, self.image_size[0], self.image_size[1], 3), dtype=jnp.uint8)
        _ = self.preprocess_batch(dummy_data)
        logger.debug("[Worker {}] Ready.", os.getpid())

    def __call__(self, batch):
        imgs_jax = jnp.array(np.stack(batch["image"]))
        proc_imgs_jax = self.preprocess_batch(imgs_jax)
        proc_imgs = np.array(proc_imgs_jax)

        vecs = self.triton.infer(proc_imgs)

        ids = batch["id"].tolist() if not self.config["COLLECTION_AUTO_ID"] else None
        self.milvus.insert(ids, vecs)

        return {"status": ["ok"] * len(vecs)}

    def __del__(self):
        if hasattr(self, "triton"):
            try:
                self.triton.close()
            except Exception:
                pass


class RayEmbeddingPipeline:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = load_config(config_path)
        self._setup_env()

    def _setup_env(self):
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = self.config["JAX_MEM_FRACTION"]
        os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

    def run(self, n_images=5000):
        logger.info("Setting up Milvus...")
        create_collection_no_index(self.config)
        logger.info("Milvus collection ready (no index).")

        if ray.is_initialized():
            ray.shutdown()
        ray.init()

        logger.info("Generating dummy data...")
        image_size = extract_image_size(self.config["INPUT_SHAPE"])
        auto_id = bool(self.config["COLLECTION_AUTO_ID"])

        data = [
            {
                "image": np.random.randint(0, 255, (image_size[0], image_size[1], 3), dtype=np.uint8),
                **({} if auto_id else {"id": i}),
            }
            for i in range(n_images)
        ]
        ds = ray.data.from_items(data)

        strategy = ActorPoolStrategy(
            min_size=self.config["WORKER_POOL_SIZE"][0],
            max_size=self.config["WORKER_POOL_SIZE"][1],
        )

        logger.info("Starting pipeline...")
        start_time = time.time()

        ds = ds.map_batches(
            Processor,
            compute=strategy,
            batch_size=self.config["BATCH_SIZE"],
            num_gpus=self.config["NUM_GPUS_PER_WORKER"],
            fn_constructor_args=(self.config,),
        )

        total_samples = 0
        for res in ds.iter_batches():
            batch_count = len(res["status"])
            total_samples += batch_count

        duration = time.time() - start_time
        tps = total_samples / duration

        logger.info("Building index...")
        build_index_and_load(self.config)

        logger.info("{}", "=" * 40)
        logger.success("Processing complete.")
        logger.info("Total samples: {}", total_samples)
        logger.info("Time taken  : {:.2f} s", duration)
        logger.info("Throughput  : {:.2f} img/s", tps)
        logger.info("{}", "=" * 40)


if __name__ == "__main__":
    default_config = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    )
    config_path = os.path.expanduser(os.environ.get("APP_CONFIG", default_config))
    app = RayEmbeddingPipeline(config_path)
    app.run()