#!/usr/bin/env python3
"""Generate a Triton YOLO ensemble repository from a source folder.

Expected source folder layout (common case):
  <source>/model/model.onnx
  <source>/labelmap/labels.txt

Supported alternative layout:
  <source>/model.onnx
  <source>/labels.txt

Generated model repository layout:
  <model_repository>/<ensemble_model_name>/
    └── config.pbtxt

  <model_repository>/<onnx_model_name>/
    ├── 1/
    │   └── model.onnx (or model.plan)
    ├── config.pbtxt
    └── labels.txt (optional, cleaned if source labels exists)

  <model_repository>/<postprocess_model_name>/
    ├── 1/
    │   └── model.py
    └── config.pbtxt
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import onnx
    from onnx import TensorProto, shape_inference
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "onnx is required. Install it first, e.g.: pip install onnx"
    ) from exc


ONNX_TO_TRITON_DTYPE: Dict[int, str] = {
    TensorProto.BOOL: "TYPE_BOOL",
    TensorProto.UINT8: "TYPE_UINT8",
    TensorProto.UINT16: "TYPE_UINT16",
    TensorProto.UINT32: "TYPE_UINT32",
    TensorProto.UINT64: "TYPE_UINT64",
    TensorProto.INT8: "TYPE_INT8",
    TensorProto.INT16: "TYPE_INT16",
    TensorProto.INT32: "TYPE_INT32",
    TensorProto.INT64: "TYPE_INT64",
    TensorProto.FLOAT16: "TYPE_FP16",
    TensorProto.FLOAT: "TYPE_FP32",
    TensorProto.DOUBLE: "TYPE_FP64",
    TensorProto.STRING: "TYPE_STRING",
    TensorProto.BFLOAT16: "TYPE_BF16",
}

# /home/haoyu/projects/ray_data/utils/unit_det/unit_postprocess.py
DEFAULT_POSTPROCESS_MODEL_PY = open(
    Path(__file__).parent / "unit" / "unit_postprocess.py",
    "r",
    encoding="utf-8",
).read()

def _find_model_file(source_dir: Path) -> Path:
    candidates = [
        source_dir / "model" / "model.onnx",
        source_dir / "model.onnx",
        source_dir / "model" / "model.plan",
        source_dir / "model.plan",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "No model file found. Expected one of: "
        "model/model.onnx, model.onnx, model/model.plan, model.plan"
    )


def _find_labels_file_optional(source_dir: Path) -> Optional[Path]:
    candidates = [
        source_dir / "labelmap" / "labels.txt",
        source_dir / "labels.txt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _parse_dims(value_info: onnx.ValueInfoProto) -> List[int]:
    tensor_type = value_info.type.tensor_type
    dims: List[int] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value") and dim.dim_value > 0:
            dims.append(int(dim.dim_value))
        else:
            dims.append(-1)
    if not dims:
        return [1]
    return dims


def _convert_tensor(value_info: onnx.ValueInfoProto) -> Dict[str, object]:
    tensor_type = value_info.type.tensor_type
    data_type = ONNX_TO_TRITON_DTYPE.get(tensor_type.elem_type, "TYPE_FP32")
    return {
        "name": value_info.name,
        "data_type": data_type,
        "dims": _parse_dims(value_info),
    }


def _load_onnx_io(onnx_path: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    model = onnx.load(str(onnx_path))
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    graph = model.graph
    initializer_names = {init.name for init in graph.initializer}

    inputs = [_convert_tensor(v) for v in graph.input if v.name not in initializer_names]
    outputs = [_convert_tensor(v) for v in graph.output]

    if not inputs:
        raise ValueError("No valid model input nodes found in ONNX graph.")
    if not outputs:
        raise ValueError("No model output nodes found in ONNX graph.")
    return inputs, outputs


def _to_pbtxt_dims(dims: Sequence[int], max_batch_size: int) -> List[int]:
    if max_batch_size > 0 and len(dims) > 1:
        return list(dims[1:])
    return list(dims)


def _format_dims(dims: Sequence[int]) -> str:
    return "[ " + ", ".join(str(d) for d in dims) + " ]"


def _format_io_block(
    items: Sequence[Dict[str, object]],
    max_batch_size: int,
) -> str:
    lines: List[str] = ["["]
    for idx, item in enumerate(items):
        dims = _to_pbtxt_dims(item["dims"], max_batch_size)
        lines.extend(
            [
                "  {",
                f"    name: \"{item['name']}\"",
                f"    data_type: {item['data_type']}",
                f"    dims: {_format_dims(dims)}",
            ]
        )
        lines.append("  }" + ("," if idx < len(items) - 1 else ""))
    lines.append("]")
    return "\n".join(lines)


def _build_onnx_config(
    model_name: str,
    inputs: Sequence[Dict[str, object]],
    outputs: Sequence[Dict[str, object]],
    max_batch_size: int,
    instance_kind: str,
    instance_count: int,
) -> str:
    config = [
        f'name: "{model_name}"',
        'platform: "onnxruntime_onnx"',
        f"max_batch_size: {max_batch_size}",
        "",
        "input " + _format_io_block(inputs, max_batch_size),
        "",
        "output " + _format_io_block(outputs, max_batch_size),
        "",
        "instance_group [",
        "  {",
        f"    kind: {instance_kind}",
        f"    count: {instance_count}",
        "  }",
        "]",
    ]
    return "\n".join(config) + "\n"


def _build_postprocess_config(
    model_name: str,
    raw_output_dtype: str,
    raw_output_dims: Sequence[int],
    model_type: str,
    score_thresh: float,
    iou_thresh: float,
    top_k: int,
    min_area: float,
) -> str:
    io_input = [
        {
            "name": "raw_output",
            "data_type": raw_output_dtype,
            "dims": list(raw_output_dims),
        },
        {
            "name": "params",
            "data_type": "TYPE_STRING",
            "dims": [1],
        }
    ]
    io_output = [
        {
            "name": "boxes",
            "data_type": "TYPE_FP32",
            "dims": [-1, 6],
        }
    ]

    config = [
        f'name: "{model_name}"',
        'backend: "python"',
        "max_batch_size: 0",
        "",
        "input " + _format_io_block(io_input, max_batch_size=0),
        "",
        "output " + _format_io_block(io_output, max_batch_size=0),
        "",
        "parameters: {",
        '  key: "model_type"',
        f'  value: {{ string_value: "{model_type}" }}',
        "}",
        "parameters: {",
        '  key: "score_thresh"',
        f'  value: {{ string_value: "{score_thresh}" }}',
        "}",
        "parameters: {",
        '  key: "iou_thresh"',
        f'  value: {{ string_value: "{iou_thresh}" }}',
        "}",
        "parameters: {",
        '  key: "top_k"',
        f'  value: {{ string_value: "{top_k}" }}',
        "}",
        "parameters: {",
        '  key: "min_area"',
        f'  value: {{ string_value: "{min_area}" }}',
        "}",
    ]
    return "\n".join(config) + "\n"


def _build_ensemble_config(
    model_name: str,
    input_name: str,
    input_dtype: str,
    input_dims: Sequence[int],
    onnx_model_name: str,
    onnx_output_name: str,
    postprocess_model_name: str,
) -> str:
    io_input = [
        {
            "name": input_name,
            "data_type": input_dtype,
            "dims": list(input_dims),
        },
        {
            "name": "params",
            "data_type": "TYPE_STRING",
            "dims": [1],
        }
    ]
    io_output = [
        {
            "name": "boxes",
            "data_type": "TYPE_FP32",
            "dims": [-1, 6],
        }
    ]

    config = [
        f'name: "{model_name}"',
        'platform: "ensemble"',
        "",
        "input " + _format_io_block(io_input, max_batch_size=0),
        "",
        "output " + _format_io_block(io_output, max_batch_size=0),
        "",
        "ensemble_scheduling {",
        "  step [",
        "    {",
        f'      model_name: "{onnx_model_name}"',
        "      model_version: -1",
        "      input_map {",
        f'        key: "{input_name}"',
        f'        value: "{input_name}"',
        "      }",
        "      output_map {",
        f'        key: "{onnx_output_name}"',
        '        value: "raw_output"',
        "      }",
        "    },",
        "    {",
        f'      model_name: "{postprocess_model_name}"',
        "      model_version: -1",
        "      input_map {",
        '        key: "raw_output"',
        '        value: "raw_output"',
        "      }",
        "      input_map {",
        '        key: "params"',
        '        value: "params"',
        "      }",
        "      output_map {",
        '        key: "boxes"',
        '        value: "boxes"',
        "      }",
        "    }",
        "  ]",
        "}",
    ]
    return "\n".join(config) + "\n"


def _clean_labels(source_labels: Path, target_labels: Path) -> int:
    cleaned: List[str] = []
    with source_labels.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            if "," in raw:
                first = raw.split(",", 1)[0].strip()
            else:
                parts = raw.split()
                first = parts[0].strip() if parts else ""
            if first:
                cleaned.append(first)

    with target_labels.open("w", encoding="utf-8") as f:
        for item in cleaned:
            f.write(f"{item}\n")
    return len(cleaned)


def _resolve_postprocess_source(postprocess_source: Optional[Path]) -> Optional[Path]:
    if postprocess_source is None:
        return None

    source = postprocess_source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Postprocess source file not found: {source}")
    return source


def _ensure_distinct_names(*model_names: str) -> None:
    if len(set(model_names)) != len(model_names):
        raise ValueError(
            "ensemble model name, onnx model name, and postprocess model name must be distinct"
        )


def generate_triton_structure(
    source_dir: Path,
    model_repository: Path,
    ensemble_model_name: str,
    onnx_model_name: Optional[str],
    postprocess_model_name: Optional[str],
    max_batch_size: int,
    instance_kind: str,
    instance_count: int,
    model_type: str,
    score_thresh: float,
    iou_thresh: float,
    top_k: int,
    min_area: float,
    postprocess_source: Optional[Path],
) -> Path:
    source_dir = source_dir.resolve()
    model_repository = model_repository.resolve()

    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source folder does not exist: {source_dir}")

    resolved_onnx_model_name = onnx_model_name or f"{ensemble_model_name}_onnx"
    resolved_postprocess_model_name = (
        postprocess_model_name or f"{ensemble_model_name}_postprocess"
    )
    _ensure_distinct_names(
        ensemble_model_name,
        resolved_onnx_model_name,
        resolved_postprocess_model_name,
    )

    model_file = _find_model_file(source_dir)
    labels_file = _find_labels_file_optional(source_dir)
    postprocess_py_source = _resolve_postprocess_source(postprocess_source)

    onnx_for_config = model_file if model_file.suffix.lower() == ".onnx" else None
    if onnx_for_config is None:
        fallback_candidates = [
            source_dir / "model" / "model.onnx",
            source_dir / "model.onnx",
        ]
        onnx_for_config = next((p for p in fallback_candidates if p.is_file()), None)
        if onnx_for_config is None:
            raise ValueError(
                "Cannot auto-generate config.pbtxt without ONNX. "
                "Please provide model/model.onnx or model.onnx in source folder."
            )

    inputs, outputs = _load_onnx_io(onnx_for_config)
    first_input = inputs[0]
    first_output = outputs[0]

    onnx_dir = model_repository / resolved_onnx_model_name
    postprocess_dir = model_repository / resolved_postprocess_model_name
    ensemble_dir = model_repository / ensemble_model_name

    for model_dir in (onnx_dir, postprocess_dir, ensemble_dir):
        if model_dir.exists():
            shutil.rmtree(model_dir)

    onnx_version_dir = onnx_dir / "1"
    postprocess_version_dir = postprocess_dir / "1"
    ensemble_version_dir = ensemble_dir / "1"
    onnx_version_dir.mkdir(parents=True, exist_ok=True)
    postprocess_version_dir.mkdir(parents=True, exist_ok=True)
    ensemble_version_dir.mkdir(parents=True, exist_ok=True)

    target_model_file = onnx_version_dir / model_file.name
    shutil.copy2(model_file, target_model_file)

    cleaned_count: Optional[int] = None
    if labels_file is not None:
        target_labels = onnx_dir / "labels.txt"
        cleaned_count = _clean_labels(labels_file, target_labels)

    target_postprocess_py = postprocess_version_dir / "model.py"
    if postprocess_py_source is None:
        target_postprocess_py.write_text(DEFAULT_POSTPROCESS_MODEL_PY, encoding="utf-8")
        postprocess_source_desc = "embedded-default(numpy-only)"
    else:
        shutil.copy2(postprocess_py_source, target_postprocess_py)
        postprocess_source_desc = str(postprocess_py_source)

    onnx_config_text = _build_onnx_config(
        model_name=resolved_onnx_model_name,
        inputs=inputs,
        outputs=outputs,
        max_batch_size=max_batch_size,
        instance_kind=instance_kind,
        instance_count=instance_count,
    )
    (onnx_dir / "config.pbtxt").write_text(onnx_config_text, encoding="utf-8")

    postprocess_config_text = _build_postprocess_config(
        model_name=resolved_postprocess_model_name,
        raw_output_dtype=str(first_output["data_type"]),
        raw_output_dims=first_output["dims"],
        model_type=model_type,
        score_thresh=score_thresh,
        iou_thresh=iou_thresh,
        top_k=top_k,
        min_area=min_area,
    )
    (postprocess_dir / "config.pbtxt").write_text(postprocess_config_text, encoding="utf-8")

    ensemble_config_text = _build_ensemble_config(
        model_name=ensemble_model_name,
        input_name=str(first_input["name"]),
        input_dtype=str(first_input["data_type"]),
        input_dims=first_input["dims"],
        onnx_model_name=resolved_onnx_model_name,
        onnx_output_name=str(first_output["name"]),
        postprocess_model_name=resolved_postprocess_model_name,
    )
    (ensemble_dir / "config.pbtxt").write_text(ensemble_config_text, encoding="utf-8")

    print(f"Generated ONNX model dir: {onnx_dir}")
    print(f"Generated postprocess model dir: {postprocess_dir}")
    print(f"Generated ensemble model dir: {ensemble_dir}")
    print(f"Copied model file to: {target_model_file}")
    print(f"Generated postprocess file at: {target_postprocess_py}")
    print(f"Postprocess source: {postprocess_source_desc}")
    print(
        "ONNX I/O selected for wiring: "
        f"input={first_input['name']} ({first_input['data_type']} {_format_dims(first_input['dims'])}), "
        f"output={first_output['name']} ({first_output['data_type']} {_format_dims(first_output['dims'])})"
    )
    if cleaned_count is not None:
        print(f"Labels cleaned at: {onnx_dir / 'labels.txt'} (rows: {cleaned_count})")
    else:
        print("Labels file not found in source folder, skipped labels.txt generation.")

    return ensemble_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Triton ensemble repository: ONNX model + Python postprocess + ensemble"
        )
    )
    parser.add_argument("source_dir", type=Path, help="Source folder containing model and optional labels")
    parser.add_argument("model_repository", type=Path, help="Target Triton model repository path")

    parser.add_argument(
        "--model-name",
        type=str,
        default="unit_ensemble",
        help="Ensemble model name (default: unit_ensemble)",
    )
    parser.add_argument(
        "--onnx-model-name",
        type=str,
        default=None,
        help="ONNX sub-model name (default: <model-name>_onnx)",
    )
    parser.add_argument(
        "--postprocess-model-name",
        type=str,
        default=None,
        help="Postprocess sub-model name (default: <model-name>_postprocess)",
    )

    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=0,
        help="ONNX sub-model max_batch_size (default: 0)",
    )
    parser.add_argument(
        "--instance-kind",
        type=str,
        default="KIND_GPU",
        choices=["KIND_GPU", "KIND_CPU", "KIND_AUTO"],
        help="ONNX sub-model Triton instance kind (default: KIND_GPU)",
    )
    parser.add_argument(
        "--instance-count",
        type=int,
        default=1,
        help="ONNX sub-model Triton instance count (default: 1)",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="yolov5",
        choices=["yolov5", "yolov8", "yolov11"],
        help="YOLO family used by postprocess (default: yolov5)",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.5,
        help="Score threshold for postprocess (default: 0.5)",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.5,
        help="NMS IoU threshold for postprocess (default: 0.5)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-K after sorting in postprocess (-1 keeps all)",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=-1,
        help="Minimum box area in postprocess (-1 disables filter)",
    )
    parser.add_argument(
        "--postprocess-source",
        type=Path,
        default=None,
        help=(
            "Python source file to copy as postprocess model.py "
            "(default: embedded numpy-only model.py)"
        ),
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.max_batch_size < 0:
        parser.error("--max-batch-size must be >= 0")
    if args.instance_count <= 0:
        parser.error("--instance-count must be > 0")
    if args.score_thresh < 0:
        parser.error("--score-thresh must be >= 0")
    if args.iou_thresh < 0:
        parser.error("--iou-thresh must be >= 0")
    if args.top_k < -1:
        parser.error("--top-k must be -1 or >= 0")
    if args.min_area < -1:
        parser.error("--min-area must be -1 or >= 0")

    try:
        generate_triton_structure(
            source_dir=args.source_dir,
            model_repository=args.model_repository,
            ensemble_model_name=args.model_name,
            onnx_model_name=args.onnx_model_name,
            postprocess_model_name=args.postprocess_model_name,
            max_batch_size=args.max_batch_size,
            instance_kind=args.instance_kind,
            instance_count=args.instance_count,
            model_type=args.model_type,
            score_thresh=args.score_thresh,
            iou_thresh=args.iou_thresh,
            top_k=args.top_k,
            min_area=args.min_area,
            postprocess_source=args.postprocess_source,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Example:
# python utils/generate_unit_triton.py data/models/20260317083337 trt_models \
#   --model-name unit_ensemble \
#   --onnx-model-name CCTH-Unit \
#   --postprocess-model-name unit_postprocess
