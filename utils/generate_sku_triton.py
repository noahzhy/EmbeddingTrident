#!/usr/bin/env python3
"""Generate Triton model repository structure from a source folder.

Expected source folder layout (common case):
  <source>/model/model.onnx
  <source>/labelmap/labels.txt

Supported alternative layout (unit model):
    <source>/model.onnx
    <source>/labels.txt
    <source>/*.json

If any top-level .json exists in source folder, it is treated as a unit model,
and Triton output config will not include label_filename.

Output layout:
  <model_repository>/<model_name>/
    ├── 1/
    │   └── model.onnx (or model.plan)
    ├── config.pbtxt
    └── labels.txt
"""

from __future__ import annotations

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


def _find_labels_file(source_dir: Path) -> Path:
    candidates = [
        source_dir / "labelmap" / "labels.txt",
        source_dir / "labels.txt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("No labels file found. Expected labelmap/labels.txt or labels.txt")


def _is_unit_model(source_dir: Path) -> bool:
    return any(path.is_file() for path in source_dir.glob("*.json"))


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
    add_label_filename: bool = False,
) -> str:
    lines: List[str] = []
    lines.append("[")
    for idx, item in enumerate(items):
        dims = _to_pbtxt_dims(item["dims"], max_batch_size)
        lines.extend([
            "  {",
            f"    name: \"{item['name']}\"",
            f"    data_type: {item['data_type']}",
            f"    dims: {_format_dims(dims)}",
        ])
        if add_label_filename:
            lines.append("    label_filename: \"labels.txt\"")
        lines.append("  }" + ("," if idx < len(items) - 1 else ""))
    lines.append("]")
    return "\n".join(lines)


def _build_config(
    model_name: str,
    inputs: Sequence[Dict[str, object]],
    outputs: Sequence[Dict[str, object]],
    max_batch_size: int,
    instance_kind: str,
    instance_count: int,
    add_output_label_filename: bool,
) -> str:
    config = [
        f'name: "{model_name}"',
        'platform: "onnxruntime_onnx"',
        f"max_batch_size: {max_batch_size}",
        "",
        "input " + _format_io_block(inputs, max_batch_size, False),
        "",
        "output " + _format_io_block(outputs, max_batch_size, add_output_label_filename),
        "",
        "instance_group [",
        "  {",
        f"    kind: {instance_kind}",
        f"    count: {instance_count}",
        "  }",
        "]",
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


def generate_triton_structure(
    source_dir: Path,
    model_repository: Path,
    model_name: str,
    max_batch_size: int,
    instance_kind: str,
    instance_count: int,
) -> Path:
    source_dir = source_dir.resolve()
    model_repository = model_repository.resolve()

    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source folder does not exist: {source_dir}")

    model_file = _find_model_file(source_dir)
    labels_file = _find_labels_file(source_dir)
    is_unit_model = _is_unit_model(source_dir)

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

    model_dir = model_repository / model_name
    version_dir = model_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)

    # Clean stale model files in version folder to avoid mixed artifacts.
    for stale in (version_dir / "model.onnx", version_dir / "model.plan"):
        if stale.exists() and stale != version_dir / model_file.name:
            stale.unlink()

    target_model_file = version_dir / model_file.name
    shutil.copy2(model_file, target_model_file)

    target_labels = model_dir / "labels.txt"
    cleaned_count = _clean_labels(labels_file, target_labels)

    target_config = model_dir / "config.pbtxt"
    config_text = _build_config(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs,
        max_batch_size=max_batch_size,
        instance_kind=instance_kind,
        instance_count=instance_count,
        add_output_label_filename=not is_unit_model,
    )
    target_config.write_text(config_text, encoding="utf-8")

    print(f"Generated Triton model at: {model_dir}")
    print(f"Model file copied to: {target_model_file}")
    print(f"Config generated at: {target_config}")
    if is_unit_model:
        print("Detected unit model metadata (.json); omitted output label_filename in config.pbtxt")
    print(f"Labels cleaned at: {target_labels} (rows: {cleaned_count})")
    return model_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy model + clean labels + generate Triton config from ONNX"
    )
    parser.add_argument("source_dir", type=Path, help="Source folder containing model/ and labelmap/")
    parser.add_argument("model_repository", type=Path, help="Target Triton model repository path")
    parser.add_argument(
        "--model-name",
        type=str,
        default="sku_model",
        help="Model directory name under model_repository (default: sku_model)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=0,
        help="Triton max_batch_size (default: 0)",
    )
    parser.add_argument(
        "--instance-kind",
        type=str,
        default="KIND_GPU",
        choices=["KIND_GPU", "KIND_CPU", "KIND_AUTO"],
        help="Triton instance kind (default: KIND_GPU)",
    )
    parser.add_argument(
        "--instance-count",
        type=int,
        default=1,
        help="Triton instance count (default: 1)",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.max_batch_size < 0:
        parser.error("--max-batch-size must be >= 0")
    if args.instance_count <= 0:
        parser.error("--instance-count must be > 0")

    try:
        generate_triton_structure(
            source_dir=args.source_dir,
            model_repository=args.model_repository,
            model_name=args.model_name,
            max_batch_size=args.max_batch_size,
            instance_kind=args.instance_kind,
            instance_count=args.instance_count,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python utils/generate_sku_triton.py data/models/20260313084531 trt_models --model-name Suntory-ES-Sku
