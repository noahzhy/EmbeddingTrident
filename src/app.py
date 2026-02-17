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


def env(k, d):
    v = os.getenv(k)
    if v is None: return d
    if isinstance(d, bool):  return v.lower() in ("1","true","yes")
    if isinstance(d, int):   return int(v)
    if isinstance(d, float): return float(v)
    if isinstance(d, tuple): return tuple(map(int, v.split(",")))
    return v


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        r = yaml.safe_load(f) or {}

    m = (r.get("models") or [{}])[0]
    i = (m.get("inputs") or [{}])[0]
    o = (m.get("outputs") or [{}])[0]
    c = (r.get("collections") or [{}])[0]
    f = c.get("fields", [])

    pk  = next((x for x in f if x.get("primary_key")), {})
    vec = next((x for x in f if x.get("type") == "FLOAT_VECTOR"), {})
    idx = next((x for x in f if x.get("index")), {})

    dim = vec.get("dim") or (o.get("shape") or [None])[-1] or 768

    return {
        # pipeline
        "BATCH_SIZE": env("BATCH_SIZE", r.get("pipelines", {}).get("batch_size", 128)),
        "NUM_GPUS_PER_WORKER": env(
            "NUM_GPUS_PER_WORKER", r.get("pipelines", {}).get("num_gpus_per_worker", 1)
        ),
        "WORKER_POOL_SIZE": env(
            "WORKER_POOL_SIZE", tuple(r.get("pipelines", {}).get("worker_pool_size", [1, 2]))
        ),
        "JAX_MEM_FRACTION": env(
            "JAX_MEM_FRACTION", r.get("pipelines", {}).get("jax_mem_fraction", 0.2)
        ),

        # services
        "TRITON_URL": env(
            "TRITON_URL",
            f"{r.get('triton_server', {}).get('host', 'localhost')}:{r.get('triton_server', {}).get('port', 8001)}",
        ),
        "MILVUS_HOST": env("MILVUS_HOST", r.get("milvus_server", {}).get("host", "localhost")),
        "MILVUS_PORT": env("MILVUS_PORT", r.get("milvus_server", {}).get("port", 19530)),

        # model
        "MODEL_NAME": env("MODEL_NAME", m.get("name", "emb_siglip2")),
        "INPUT_NAME": env("INPUT_NAME", i.get("name", "pixel_values")),
        "INPUT_DTYPE": env("INPUT_DTYPE", i.get("dtype", "FP32")),
        "INPUT_SHAPE": env("INPUT_SHAPE", tuple(i.get("shape") or [-1, 3, 224, 224])),
        "OUTPUT_NAME": env("OUTPUT_NAME", o.get("name", "image_embeds")),
        "OUTPUT_DTYPE": env("OUTPUT_DTYPE", o.get("dtype", "FP32")),

        # collection
        "COLLECTION_NAME": env("COLLECTION_NAME", c.get("name", "test_collection")),
        "COLLECTION_VECTOR_DIM": env("VECTOR_DIM", dim),
        "COLLECTION_AUTO_ID": env("COLLECTION_AUTO_ID", pk.get("auto_id", False)),
        "COLLECTION_INDEX_FIELD": idx.get("name"),
        "COLLECTION_INDEX_CFG": idx.get("index"),
        "COLLECTION_FIELDS": f,
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
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(self.config["JAX_MEM_FRACTION"])
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