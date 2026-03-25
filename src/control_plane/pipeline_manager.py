"""Pipeline lifecycle manager.

Orchestrates: model download → Triton config generation → Triton hot-load →
Ray Serve pipeline build & deploy.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ray import serve

from src.control_plane.triton_manager import TritonModelManager
from utils.blob_manager import BlobManager, RETAIL_MLMODELS_CONNECTION_STRING

# Lazy imports — these are heavyweight and only needed during build_and_deploy.
_generate_unit = None
_generate_sku = None


def _get_generate_unit():
    global _generate_unit
    if _generate_unit is None:
        from utils.generate_unit_triton import generate_triton_structure
        _generate_unit = generate_triton_structure
    return _generate_unit


def _get_generate_sku():
    global _generate_sku
    if _generate_sku is None:
        from utils.generate_sku_triton import generate_triton_structure
        _generate_sku = generate_triton_structure
    return _generate_sku


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_TRITON_HTTP = os.getenv("TRITON_HTTP_URL", "http://localhost:8000")
_DEFAULT_TRITON_GRPC_HOST = os.getenv("TRITON_GRPC_HOST", "localhost")
_DEFAULT_TRITON_GRPC_PORT = int(os.getenv("TRITON_GRPC_PORT", "8001"))
_DEFAULT_MODEL_REPO = Path(os.getenv("TRITON_MODEL_REPO", "trt_models"))
_DEFAULT_DOWNLOAD_DIR = Path(os.getenv("MODEL_DOWNLOAD_DIR", "downloaded_models"))
_ACTIVE_CONFIG_PATH = Path(os.getenv("ACTIVE_PIPELINE_CONFIG", "config/active_pipeline.json"))

# Triton model-repo is container-mounted at /models but the host path is .active_models
_ACTIVE_MODELS_DIR = Path(os.getenv("ACTIVE_MODELS_DIR", ".active_models"))


def _infer_pipeline_type(models: List[Dict]) -> str:
    types = {m["model_type"] for m in models}
    has_det = "Object_Detection" in types
    has_cls = "Classification" in types
    if has_det and has_cls:
        return "unit_sku"
    if has_det:
        return "unit"
    if has_cls:
        return "sku"
    raise ValueError(f"Cannot infer pipeline type from model_types: {types}")


class PipelineManager:
    """Manages the full lifecycle of downloading, configuring, loading, and
    deploying a model pipeline."""

    def __init__(
        self,
        triton_http_url: str = _DEFAULT_TRITON_HTTP,
        triton_grpc_host: str = _DEFAULT_TRITON_GRPC_HOST,
        triton_grpc_port: int = _DEFAULT_TRITON_GRPC_PORT,
        model_repo: Path = _DEFAULT_MODEL_REPO,
        download_dir: Path = _DEFAULT_DOWNLOAD_DIR,
        active_models_dir: Path = _ACTIVE_MODELS_DIR,
        config_path: Path = _ACTIVE_CONFIG_PATH,
    ):
        self.triton = TritonModelManager(triton_http_url)
        self.triton_grpc_host = triton_grpc_host
        self.triton_grpc_port = triton_grpc_port
        self.model_repo = Path(model_repo)
        self.download_dir = Path(download_dir)
        self.active_models_dir = Path(active_models_dir)
        self.config_path = Path(config_path)
        self._current_config: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Core orchestration
    # ------------------------------------------------------------------

    def build_and_deploy(self, models_config: List[Dict]) -> Dict[str, Any]:
        """Full pipeline:
        1. Download models from blob storage
        2. Generate Triton configs
        3. Copy to active model repo & hot-load into Triton
        4. Build & deploy Ray Serve pipeline
        5. Persist config
        """
        pipeline_type = _infer_pipeline_type(models_config)
        unit_model_name: Optional[str] = None
        sku_model_name: Optional[str] = None
        loaded_models: List[str] = []

        for model_cfg in models_config:
            model_type = model_cfg["model_type"]
            model_name = model_cfg["model_name"]
            timestamp = model_cfg["timestamp"]

            # Step 1: download (skip if already cached locally)
            expected_dir = self.download_dir / model_type / model_name / timestamp
            if expected_dir.is_dir() and any(expected_dir.rglob("*.onnx")):
                print(f"[pipeline] using cached {expected_dir}")
                source_dir = expected_dir
            else:
                print(f"[pipeline] downloading {model_type}/{model_name}/{timestamp} ...")
                blob_mgr = BlobManager(RETAIL_MLMODELS_CONNECTION_STRING)
                download_root = blob_mgr.download_model_folder(
                    model_type=model_type,
                    key_strings=[model_name, timestamp],
                    local_dir=str(self.download_dir),
                )
                source_dir = Path(download_root)
                print(f"[pipeline] downloaded to {source_dir}")

            # Step 2: generate Triton config
            self.model_repo.mkdir(parents=True, exist_ok=True)
            if model_type == "Object_Detection":
                onnx_sub_name = f"_{model_name}"
                postprocess_sub_name = "unit_postprocess"
                _get_generate_unit()(
                    source_dir=source_dir,
                    model_repository=self.model_repo,
                    ensemble_model_name=model_name,
                    onnx_model_name=onnx_sub_name,
                    postprocess_model_name=postprocess_sub_name,
                    max_batch_size=0,
                    instance_kind="KIND_GPU",
                    instance_count=1,
                    model_type="yolov5",
                    score_thresh=0.5,
                    iou_thresh=0.5,
                    top_k=-1,
                    min_area=-1,
                    postprocess_source=None,
                )
                unit_model_name = model_name
                triton_model_names = [model_name, onnx_sub_name, postprocess_sub_name]
            elif model_type == "Classification":
                _get_generate_sku()(
                    source_dir=source_dir,
                    model_repository=self.model_repo,
                    model_name=model_name,
                    max_batch_size=0,
                    instance_kind="KIND_GPU",
                    instance_count=1,
                )
                sku_model_name = model_name
                triton_model_names = [model_name]
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            # Step 3: copy generated dirs into the active model repo and load
            self.active_models_dir.mkdir(parents=True, exist_ok=True)
            import shutil

            def _ignore_pycache(directory, contents):
                return [c for c in contents if c == "__pycache__"]

            for tname in triton_model_names:
                src = self.model_repo / tname
                dst = self.active_models_dir / tname
                if dst.exists():
                    shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(src, dst, ignore=_ignore_pycache, dirs_exist_ok=True)

            for tname in triton_model_names:
                print(f"[pipeline] loading Triton model: {tname}")
                self.triton.load_model(tname)
                loaded_models.append(tname)

        # Step 3b: wait for all models to be ready
        for tname in loaded_models:
            ok = self.triton.wait_model_ready(tname, timeout=120)
            if not ok:
                raise RuntimeError(f"Triton model '{tname}' did not become ready within 120s")
            print(f"[pipeline] Triton model ready: {tname}")

        # Step 4: build & deploy Ray Serve pipelines
        from src.main import build_pipelines
        pipelines = build_pipelines(
            pipeline_type=pipeline_type,
            unit_model_name=unit_model_name,
            sku_model_name=sku_model_name,
            triton_host=self.triton_grpc_host,
            triton_port=self.triton_grpc_port,
        )

        sub_services = []
        for name, pipeline in pipelines.items():
            sub_services.append(
                serve.RunTarget(
                    target=pipeline,
                    name=name,
                    route_prefix=f"/{name}",
                )
            )
            print(f"[pipeline] deploying pipeline '{name}' at /{name}")

        serve.run_many(sub_services)
        print(f"[pipeline] deployed {list(pipelines.keys())}")

        # Step 5: persist config
        config = {
            "models": models_config,
            "pipeline_type": pipeline_type,
            "unit_model_name": unit_model_name,
            "sku_model_name": sku_model_name,
            "deployed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_config(config)
        self._current_config = config

        return config

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        config = self._current_config or self._load_config()
        triton_ready = self.triton.is_server_ready()
        loaded = self.triton.get_loaded_model_names() if triton_ready else []
        return {
            "active_config": config,
            "triton_server_ready": triton_ready,
            "triton_loaded_models": loaded,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_config(self, config: Dict) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[pipeline] config saved to {self.config_path}")

    def _load_config(self) -> Optional[Dict]:
        if not self.config_path.is_file():
            return None
        try:
            return json.loads(self.config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def restore_from_config(self) -> Optional[Dict]:
        """Restore pipeline from persisted config (called at startup)."""
        config = self._load_config()
        if config is None:
            print("[pipeline] no saved config found, starting empty")
            return None
        print(f"[pipeline] restoring from saved config: {config}")
        return self.build_and_deploy(config["models"])
