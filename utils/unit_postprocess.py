import json
import cupy as cp
import triton_python_backend_utils as pb_utils
import concurrent.futures


def nms_cupy(boxes, scores, iou_thresh):
    if boxes.shape[0] == 0:
        return cp.empty((0,), dtype=cp.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = cp.maximum(0.0, x2 - x1) * cp.maximum(0.0, y2 - y1)
    order = cp.argsort(scores)[::-1]

    keep = []
    while order.size > 0:
        i = order[0].item() 
        keep.append(i)
        if order.size == 1:
            break

        xx1 = cp.maximum(x1[i], x1[order[1:]])
        yy1 = cp.maximum(y1[i], y1[order[1:]])
        xx2 = cp.minimum(x2[i], x2[order[1:]])
        yy2 = cp.minimum(y2[i], y2[order[1:]])

        w = cp.maximum(0.0, xx2 - xx1)
        h = cp.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = cp.where(union > 0.0, inter / union, 0.0)

        remain = cp.where(iou <= iou_thresh)[0]
        order = order[remain + 1]

    return cp.asarray(keep, dtype=cp.int64)


class TritonPythonModel:

    def initialize(self, args):
        model_config = args.get("model_config", {})
        if isinstance(model_config, str):
            model_config = json.loads(model_config)

        params = model_config.get("parameters", {})

        def _get_param(name, default):
            value = params.get(name)
            if value is None:
                return default
            if isinstance(value, dict):
                return value.get("string_value", default)
            return value

        self.model_type = str(_get_param("model_type", "yolov5"))
        self.score_thresh = float(_get_param("score_thresh", 0.5))
        self.iou_thresh = float(_get_param("iou_thresh", 0.5))
        self.top_k = int(_get_param("top_k", -1))
        self.min_area = float(_get_param("min_area", -1))
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

    def _to_float_pair(self, value):
        if value is None:
            return None

        if isinstance(value, dict):
            if "h" in value and "w" in value:
                return float(value["h"]), float(value["w"])
            if "height" in value and "width" in value:
                return float(value["height"]), float(value["width"])
            return None

        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return float(value[0]), float(value[1])

        return None

    def _to_pad_info(self, value):
        if value is None:
            return None

        if isinstance(value, dict):
            if all(k in value for k in ("pad_w", "pad_h", "scale")):
                return (
                    float(value["pad_w"]),
                    float(value["pad_h"]),
                    float(value["scale"]),
                )
            return None

        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return float(value[0]), float(value[1]), float(value[2])

        return None

    def _resolve_runtime_params(self, request):
        runtime = {
            "model_type": self.model_type,
            "input_shape": None,
            "orig_shape": None,
            "pad_info": None,
        }

        params_tensor = pb_utils.get_input_tensor_by_name(request, "params")
        if params_tensor is None:
            return runtime

        try:
            params_raw = params_tensor.as_numpy()[0]
            if isinstance(params_raw, (bytes, bytearray)):
                params_raw = params_raw.decode("utf-8")
            params = json.loads(params_raw)

            runtime["model_type"] = str(params.get("model_type", runtime["model_type"]))

            input_shape = self._to_float_pair(params.get("input_shape"))
            if input_shape is None:
                input_h = params.get("input_shape_h")
                input_w = params.get("input_shape_w")
                if input_h is not None and input_w is not None:
                    input_shape = float(input_h), float(input_w)
            runtime["input_shape"] = input_shape

            orig_shape = self._to_float_pair(params.get("orig_shape"))
            if orig_shape is None:
                orig_h = params.get("orig_shape_h")
                orig_w = params.get("orig_shape_w")
                if orig_h is not None and orig_w is not None:
                    orig_shape = float(orig_h), float(orig_w)
            runtime["orig_shape"] = orig_shape

            pad_info = self._to_pad_info(params.get("pad_info"))
            if pad_info is None:
                pad_w = params.get("pad_w")
                pad_h = params.get("pad_h")
                scale = params.get("scale")
                if pad_w is not None and pad_h is not None and scale is not None:
                    pad_info = float(pad_w), float(pad_h), float(scale)
            runtime["pad_info"] = pad_info

            return runtime
        except Exception:
            return runtime

    def normalize_to_original(self, boxes, runtime):
        if boxes.shape[0] == 0:
            return boxes.astype(cp.float32)

        orig_shape = runtime.get("orig_shape")
        pad_info = runtime.get("pad_info")

        if orig_shape is not None and pad_info is not None:
            orig_h, orig_w = orig_shape
            pad_w, pad_h, scale = pad_info

            if orig_h > 0 and orig_w > 0 and scale > 0:
                boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale

                cp.clip(boxes[:, [0, 2]], 0.0, orig_w, out=boxes[:, [0, 2]])
                cp.clip(boxes[:, [1, 3]], 0.0, orig_h, out=boxes[:, [1, 3]])

                boxes[:, [0, 2]] /= orig_w
                boxes[:, [1, 3]] /= orig_h

                cp.round(boxes[:, :4], decimals=6, out=boxes[:, :4])
                cp.clip(boxes[:, :4], 0.0, 1.0, out=boxes[:, :4])
                return boxes.astype(cp.float32)

        input_shape = runtime.get("input_shape")
        if input_shape is not None:
            input_h, input_w = input_shape
            if input_h > 0 and input_w > 0:
                boxes[:, [0, 2]] /= input_w
                boxes[:, [1, 3]] /= input_h
                cp.round(boxes[:, :4], decimals=6, out=boxes[:, :4])
                cp.clip(boxes[:, :4], 0.0, 1.0, out=boxes[:, :4])
                return boxes.astype(cp.float32)

        return boxes.astype(cp.float32)

    def process_single_request(self, request):
        in_tensor = pb_utils.get_input_tensor_by_name(request, "raw_output")

        if in_tensor.is_cpu():
            raw = cp.asarray(in_tensor.as_numpy())
        else:
            raw = cp.from_dlpack(in_tensor.to_dlpack())

        runtime = self._resolve_runtime_params(request)
        boxes = self.decode(raw, runtime["model_type"])

        if boxes.shape[0] == 0:
            out = cp.zeros((0, 6), dtype=cp.float32)
        else:
            boxes = self.postprocess(boxes)
            out = self.normalize_to_original(boxes, runtime)

        out_tensor = pb_utils.Tensor.from_dlpack("boxes", out.toDlpack())
        return pb_utils.InferenceResponse(output_tensors=[out_tensor])

    def execute(self, requests):
        responses = list(self.executor.map(self.process_single_request, requests))
        return responses

    def finalize(self):
        self.executor.shutdown(wait=True)

    def decode(self, output, model_type):
        if output.ndim > 0 and output.shape[0] == 1:
            output = cp.squeeze(output, axis=0)

        if model_type == "yolov5":
            return self.decode_yolov5(output)
        if model_type == "yolov8":
            return self.decode_yolov8(output)
        if model_type == "yolov11":
            return self.decode_yolov11(output)
        raise ValueError(f"Unsupported model_type: {model_type}")

    def decode_yolov5(self, pred):
        boxes = pred[:, :4]
        obj = pred[:, 4:5]
        cls = pred[:, 5:]

        obj_mask = (obj[:, 0] > self.score_thresh)
        boxes = boxes[obj_mask]
        obj = obj[obj_mask]
        cls = cls[obj_mask]

        if boxes.shape[0] == 0:
            return cp.zeros((0, 6), dtype=cp.float32)

        scores = obj * cls
        max_scores = cp.max(scores, axis=1)
        class_ids = cp.argmax(scores, axis=1)
        
        mask = max_scores > self.score_thresh
        boxes = boxes[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        boxes = self.xywh2xyxy(boxes)
        return self.pack(boxes, max_scores, class_ids)

    def decode_yolov8(self, pred):
        boxes = pred[:, :4]
        cls = pred[:, 4:]

        scores = cp.max(cls, axis=1)
        mask = scores > self.score_thresh

        boxes = boxes[mask]
        filtered_cls = cls[mask]
        scores = scores[mask]
        
        if boxes.shape[0] == 0:
            return cp.zeros((0, 6), dtype=cp.float32)

        class_ids = cp.argmax(filtered_cls, axis=1)

        boxes = self.xywh2xyxy(boxes)
        return self.pack(boxes, scores, class_ids)

    def decode_yolov11(self, pred):
        return self.decode_yolov8(pred)

    def xywh2xyxy(self, x):
        y = cp.empty_like(x)
        w_half = x[:, 2] / 2
        h_half = x[:, 3] / 2
        y[:, 0] = x[:, 0] - w_half
        y[:, 1] = x[:, 1] - h_half
        y[:, 2] = x[:, 0] + w_half
        y[:, 3] = x[:, 1] + h_half
        y = cp.round(y, decimals=6)
        return y

    def pack(self, boxes, scores, class_ids):
        return cp.concatenate(
            [boxes, scores[:, None], class_ids[:, None]], axis=1
        )

    def postprocess(self, boxes):
        if boxes.shape[0] == 0:
            return boxes.astype(cp.float32)

        keep = nms_cupy(boxes[:, :4], boxes[:, 4], self.iou_thresh)
        b = boxes[keep]

        if self.min_area >= 0:
            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            b = b[area > self.min_area]
            if b.shape[0] == 0:
                return cp.zeros((0, 6), dtype=cp.float32)

        scores = b[:, 4]
        idx = cp.argsort(scores)[::-1]
        b = b[idx]

        if self.top_k >= 0:
            b = b[:self.top_k]

        return b.astype(cp.float32)