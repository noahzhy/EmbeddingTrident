#!/usr/bin/env python3
import base64
import io
import json
import os
import socket
import tempfile
import time
import uuid
from typing import Dict, List, Optional, Tuple

import gradio as gr
import requests
from PIL import Image


DEFAULT_APP = "unit_sku"
REQUEST_TIMEOUT = 15


def _save_uploaded_image(uploaded_image: Optional[Image.Image]) -> str:
    if uploaded_image is None:
        raise RuntimeError("请先上传图片")

    upload_dir = os.path.join(tempfile.gettempdir(), "ray_data_demo_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    filename = f"upload_{int(time.time())}_{uuid.uuid4().hex[:8]}.jpg"
    file_path = os.path.join(upload_dir, filename)
    uploaded_image.convert("RGB").save(file_path, format="JPEG", quality=95)
    return file_path


def use_uploaded_image(uploaded_image: Optional[Image.Image]) -> str:
    image_path = _save_uploaded_image(uploaded_image)
    return image_path


def _get_host_ip() -> str:
    """Best-effort host IP detection to mimic shell fallback candidates."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return ""


def _build_candidates(
    override_endpoint: str,
    env_name: str,
    route: str,
) -> List[str]:
    endpoint = (override_endpoint or "").strip() or os.getenv(env_name, "").strip()
    if endpoint:
        return [endpoint]

    candidates = [
        # f"http://0.0.0.0:2866/{route}",
        f"http://127.0.0.1:2866/{route}",
    ]
    host_ip = _get_host_ip()
    if host_ip:
        candidates.append(f"http://{host_ip}:2866/{route}")
    return candidates


def _post_with_fallback(candidates: List[str], payload: Dict) -> Tuple[str, int, str]:
    last_status = 0
    last_body = ""

    for candidate in candidates:
        try:
            resp = requests.post(candidate, json=payload, timeout=REQUEST_TIMEOUT)
            status = resp.status_code
            body = resp.text
        except requests.RequestException as exc:
            status = 0
            body = str(exc)

        if 200 <= status < 300:
            return candidate, status, body

        last_status = status
        last_body = body

    raise RuntimeError(
        f"request failed: http_status={last_status}, response={last_body[:500]}"
    )


def _parse_json(body: str) -> Dict:
    body = (body or "").strip()
    if not body:
        raise RuntimeError("request failed: empty response body")
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        preview = body[:400].replace("\n", "\\n")
        raise RuntimeError(f"request failed: non-json response: {preview}") from exc


def request_unit_sku(
    image_url: str,
    app: str,
    unit_endpoint_override: str,
) -> str:
    payload = {"image_url": image_url.strip(), "app": app.strip() or DEFAULT_APP}
    candidates = _build_candidates(unit_endpoint_override, "UNIT_ENDPOINT", "unit_sku")
    used_endpoint, status, body = _post_with_fallback(candidates, payload)
    data = _parse_json(body)

    wrapped = {
        "used_endpoint": used_endpoint,
        "http_status": status,
        "payload": payload,
        "response": data,
    }
    return json.dumps(wrapped, ensure_ascii=False, indent=2)


def request_visual(
    image_url: str,
    app: str,
    visual_endpoint_override: str,
) -> Tuple[Optional[Image.Image], str]:
    payload = {"image_url": image_url.strip(), "app": app.strip() or DEFAULT_APP}
    candidates = _build_candidates(visual_endpoint_override, "VISUAL_ENDPOINT", "visual")
    used_endpoint, status, body = _post_with_fallback(candidates, payload)
    data = _parse_json(body)

    if "error" in data:
        raise RuntimeError(f"request failed: {data['error']}")

    image_b64 = data.get("image_base64", "")
    image_obj: Optional[Image.Image] = None
    if image_b64:
        try:
            image_obj = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"decode image_base64 failed: {exc}") from exc

    wrapped = {
        "used_endpoint": used_endpoint,
        "http_status": status,
        "payload": payload,
        "visual_source": data.get("visual_source"),
        "response": data,
    }
    return image_obj, json.dumps(wrapped, ensure_ascii=False, indent=2)


def run_both(
    image_url: str,
    app: str,
    unit_endpoint_override: str,
    visual_endpoint_override: str,
) -> Tuple[Optional[Image.Image], str, str]:
    unit_json = request_unit_sku(image_url, app, unit_endpoint_override)
    image_obj, visual_json = request_visual(image_url, app, visual_endpoint_override)
    return image_obj, unit_json, visual_json


def build_ui() -> gr.Blocks:
    custom_css = """
    #control-panel, #result-panel {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 14px;
        background: #fafafa;
    }
    #action-row button {
        min-height: 44px;
    }
    #visual-image {
        height: min(70vh, 720px) !important;
    }
    #visual-image img {
        object-fit: contain !important;
    }
    @media (max-width: 900px) {
        #visual-image {
            height: 52vh !important;
        }
    }
    """

    with gr.Blocks(title="Ray Unit/SKU Visual Demo", css=custom_css) as demo:
        gr.Markdown("# Ray Unit/SKU Visual Demo")
        gr.Markdown(
            "上传或填写图片路径后，可分别请求 `/unit_sku` 与 `/visual`，右侧查看可视化与 JSON 结果。"
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_id="control-panel"):
                gr.Markdown("### Input")
                upload_image = gr.Image(
                    label="上传图片",
                    type="pil",
                )

                with gr.Row():
                    image_url = gr.Textbox(
                        label="image_url",
                        value="data/images/unit_test.jpg",
                        placeholder="后端可访问的本地路径或 HTTP URL",
                        scale=1,
                    )

                with gr.Row():
                    app = gr.Dropdown(
                        label="app",
                        choices=["unit_sku", "unit", "sku"],
                        value=DEFAULT_APP,
                        allow_custom_value=True,
                    )

                with gr.Accordion("Endpoint 覆盖（可选）", open=False):
                    unit_endpoint_override = gr.Textbox(
                        label="UNIT endpoint",
                        placeholder="e.g. http://127.0.0.1:2866/unit_sku",
                    )
                    visual_endpoint_override = gr.Textbox(
                        label="VISUAL endpoint",
                        placeholder="e.g. http://127.0.0.1:2866/visual",
                    )

                gr.Markdown("### Actions")
                with gr.Row(elem_id="action-row"):
                    btn_unit   = gr.Button("获取返回JSON",  variant="secondary")
                    btn_visual = gr.Button("获取可视化图像", variant="secondary")
                    btn_both   = gr.Button("同时获取",      variant="primary")

            with gr.Column(scale=7, elem_id="result-panel"):
                gr.Markdown("### Results")
                with gr.Tabs():
                    with gr.Tab("可视化结果"):
                        out_image = gr.Image(
                            label="可视化结果",
                            type="pil",
                            height=560,
                            elem_id="visual-image",
                        )

                    with gr.Tab("unit_sku response"):
                        out_unit_json = gr.Code(label="unit_sku JSON", language="json")

                    with gr.Tab("visual response"):
                        out_visual_json = gr.Code(label="visual JSON", language="json")

        btn_unit.click(
            fn=request_unit_sku,
            inputs=[image_url, app, unit_endpoint_override],
            outputs=[out_unit_json],
        )
        upload_image.change(
            fn=use_uploaded_image,
            inputs=[upload_image],
            outputs=[image_url],
        )
        btn_visual.click(
            fn=request_visual,
            inputs=[image_url, app, visual_endpoint_override],
            outputs=[out_image, out_visual_json],
        )
        btn_both.click(
            fn=run_both,
            inputs=[image_url, app, unit_endpoint_override, visual_endpoint_override],
            outputs=[out_image, out_unit_json, out_visual_json],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860)
