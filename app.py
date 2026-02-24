"""
CV Transfer Learning + Grad-CAM Demo
Interactive Gradio app for image classification with visual explanations.
Supports PyTorch & TensorFlow backends.
"""
import os
import time
import numpy as np

# ── Patch gradio_client bug BEFORE importing gradio ─────────────────────
# gradio_client's json_schema_to_python_type crashes when a JSON-schema
# value is a plain bool (e.g. additionalProperties: true).  We wrap the
# two affected helpers so they return a safe fallback instead of crashing.
import gradio_client.utils as _gc_utils  # noqa: E402

_orig_get_type = _gc_utils.get_type
_orig_json_to_py = _gc_utils._json_schema_to_python_type


def _safe_get_type(schema):
    if isinstance(schema, bool):
        return "bool"
    return _orig_get_type(schema)


def _safe_json_to_py(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_json_to_py(schema, defs)


_gc_utils.get_type = _safe_get_type
_gc_utils._json_schema_to_python_type = _safe_json_to_py
# ── End patch ───────────────────────────────────────────────────────────

import gradio as gr
from PIL import Image

# ── Lazy framework imports ──────────────────────────────────────────────
try:
    import torch
    from src.pytorch_transfer import PyTorchTransferModel
except ImportError:
    torch = None
    PyTorchTransferModel = None

try:
    from src.tensorflow_transfer import TensorFlowTransferModel
except ImportError:
    TensorFlowTransferModel = None

from src.gradcam import GradCAM

# ── Model cache ────────────────────────────────────────────────────────
_MODEL_CACHE = {}


def _get_model(framework: str, model_name: str):
    key = f"{framework}_{model_name}"
    if key not in _MODEL_CACHE:
        if framework == "PyTorch":
            _MODEL_CACHE[key] = PyTorchTransferModel(
                model_name=model_name.lower(), num_classes=1000)
        else:
            _MODEL_CACHE[key] = TensorFlowTransferModel(
                model_name=model_name, num_classes=1000)
    return _MODEL_CACHE[key]


# ── Main prediction function ───────────────────────────────────────────

def predict_with_gradcam(image, framework, model_name):
    """Classify image → top-5 predictions + Grad-CAM overlays."""
    if image is None:
        return "", None, None, "Upload an image and click **Classify**."

    try:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        if framework == "PyTorch" and PyTorchTransferModel is None:
            return ("", None, None,
                    "❌ PyTorch not installed (`pip install torch torchvision`)")
        if framework == "TensorFlow" and TensorFlowTransferModel is None:
            return ("", None, None,
                    "❌ TensorFlow not installed (`pip install tensorflow`)")

        t0 = time.time()
        model = _get_model(framework, model_name)
        results = model.predict(image, top_k=5)
        elapsed = time.time() - t0

        # Build HTML bar chart for predictions
        label_html = _build_label_html(results)

        # Grad-CAM
        fw = "pytorch" if framework == "PyTorch" else "tensorflow"
        gradcam = GradCAM(model.model, framework=fw,
                          target_layer=model.get_target_layer_name())

        heatmap = gradcam.generate_heatmap(image, target_class=None)
        overlay = gradcam.overlay_heatmap(image, heatmap, alpha=0.45)

        if fw == "pytorch":
            gradcam.remove_hooks()

        top_name, top_conf = results[0]
        info = (f"**{top_name}** — {top_conf:.1%} confidence\n\n"
                f"`{framework}` · `{model_name}` · "
                f"{elapsed:.2f}s · {image.size[0]}×{image.size[1]}px")

        return label_html, overlay, heatmap, info

    except Exception as e:
        return "", None, None, f"❌ Error: {e}"


def _build_label_html(results):
    """Build a polished HTML bar chart from prediction results."""
    if not results:
        return ""
    max_conf = max(conf for _, conf in results)
    rows = []
    for i, (name, conf) in enumerate(results):
        pct = conf * 100
        bar_w = (conf / max_conf * 100) if max_conf > 0 else 0
        # Top result gets accent color, others get dimmer tone
        if i == 0:
            bar_bg = "linear-gradient(90deg,#f97316 0%,#fb923c 100%)"
            rank_bg = "#f97316"
            name_color = "#ffffff"
            pct_color = "#f97316"
        else:
            bar_bg = "linear-gradient(90deg,#3a3a3a 0%,#4a4a4a 100%)"
            rank_bg = "#333"
            name_color = "#d4d4d4"
            pct_color = "#a1a1a1"
        rows.append(
            f'<div style="display:flex;align-items:center;gap:10px;'
            f'padding:6px 0;'
            f'border-bottom:1px solid rgba(255,255,255,.04);">'
            f'<span style="min-width:22px;height:22px;display:flex;'
            f'align-items:center;justify-content:center;font-size:.7rem;'
            f'font-weight:700;color:#fff;background:{rank_bg};'
            f'border-radius:6px;">{i+1}</span>'
            f'<span style="min-width:140px;max-width:180px;font-size:.84rem;'
            f'color:{name_color};white-space:nowrap;overflow:hidden;'
            f'text-overflow:ellipsis;font-weight:{"600" if i==0 else "400"};">'
            f'{name}</span>'
            f'<div style="flex:1;background:#1a1a1a;border-radius:8px;'
            f'height:10px;overflow:hidden;">'
            f'<div style="width:{bar_w:.1f}%;height:100%;{bar_bg};'
            f'border-radius:8px;transition:width .5s cubic-bezier(.4,0,.2,1);">'
            f'</div></div>'
            f'<span style="min-width:52px;font-size:.82rem;color:{pct_color};'
            f'text-align:right;font-weight:600;font-variant-numeric:tabular-nums;">'
            f'{pct:.1f}%</span>'
            f'</div>'
        )
    return (
        f'<div style="padding:4px 0;">'
        f'<div style="display:flex;align-items:center;gap:6px;'
        f'margin-bottom:10px;">'
        f'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" '
        f'stroke="#f97316" stroke-width="2.5" stroke-linecap="round">'
        f'<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>'
        f'<polyline points="22 4 12 14.01 9 11.01"/></svg>'
        f'<span style="font-size:.75rem;color:#737373;font-weight:600;'
        f'text-transform:uppercase;letter-spacing:.06em;">Predictions</span>'
        f'</div>'
        f'{"".join(rows)}</div>'
    )


# ── CSS ────────────────────────────────────────────────────────────────

CSS = """
/* ── Root variables ── */
:root {
    --accent: #f97316;
    --accent-hover: #ea580c;
    --accent-soft: rgba(249,115,22,.08);
    --surface: #0f0f0f;
    --surface-2: #1a1a1a;
    --surface-3: #242424;
    --border: #2e2e2e;
    --text: #f1f1f1;
    --text-muted: #a1a1a1;
    --radius: 14px;
    --shadow-sm: 0 1px 3px rgba(0,0,0,.3);
    --shadow-md: 0 4px 16px rgba(0,0,0,.4);
}

/* ── Hide Gradio footer & scrollbar flicker ── */
footer { display: none !important; }
.gradio-container {
    max-width: 1100px !important;
    margin: auto;
    background: var(--surface) !important;
}
.dark, body, .main, .app {
    background: var(--surface) !important;
}

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 28px 16px 12px;
}
.app-header h1 {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--accent);
    margin: 0 0 4px;
    letter-spacing: -.02em;
}
.app-header p {
    font-size: .92rem;
    color: var(--text-muted);
    margin: 0;
}
.app-header .badge-row {
    display: flex;
    justify-content: center;
    gap: 6px;
    margin-top: 10px;
    flex-wrap: wrap;
}
.app-header .badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: .72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: .01em;
}
.badge.pytorch  { background: #ee4c2c22; color: #ff6b5b; border: 1px solid #ee4c2c44; }
.badge.tf       { background: #ff6f0022; color: #ffaa60; border: 1px solid #ff6f0044; }
.badge.gradio   { background: #f9731622; color: #f97316; border: 1px solid #f9731644; }
.badge.imagenet { background: #f9731622; color: #fbbf24; border: 1px solid #fbbf2444; }

/* ── Cards ── */
.card {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    box-shadow: var(--shadow-sm);
}

/* ── Predict button — FIXED SIZE ── */
#predict-btn {
    width: 100% !important;
    min-height: 48px !important;
    max-height: 48px !important;
    height: 48px !important;
    font-size: .95rem !important;
    font-weight: 600 !important;
    letter-spacing: .01em;
    border-radius: 10px !important;
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    cursor: pointer;
    transition: background .2s, box-shadow .2s, transform .1s;
    box-shadow: 0 2px 8px rgba(249,115,22,.3);
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: hidden !important;
    flex-shrink: 0 !important;
    line-height: 1 !important;
    padding: 0 16px !important;
    box-sizing: border-box !important;
}
#predict-btn:hover {
    background: var(--accent-hover) !important;
    box-shadow: 0 4px 14px rgba(249,115,22,.45);
    transform: translateY(-1px);
}
#predict-btn:active {
    transform: translateY(0);
    box-shadow: 0 1px 4px rgba(249,115,22,.2);
}

/* ── Kill extra spinners — show only ONE loader ── */
#predict-btn .wrap,
#predict-btn .loading,
.results-col .progress-bar,
.results-col .wrap.default,
.results-col > div > .wrap.default,
#overlay-img .wrap,
#heatmap-img .wrap,
#info-box .wrap {
    display: none !important;
}
/* Only the results area keeps its subtle loader */

/* ── Label (predictions) ── */
#label-out {
    position: relative;
    min-height: 100px;
    background: var(--surface-2);
    border-radius: var(--radius);
    padding: 12px 16px;
    border: 1px solid var(--border);
}

/* ── Dropdown selects ── */
.settings-row .gr-dropdown {
    border-radius: 10px !important;
}

/* ── Image upload area ── */
#img-upload {
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    transition: border-color .2s;
    min-height: 260px;
}
#img-upload:hover { border-color: var(--accent) !important; }

/* ── Result info ── */
#info-box {
    min-height: 48px;
    padding: 10px 14px;
    background: var(--surface-3);
    border-radius: 10px;
    border: 1px solid var(--border);
}
#info-box p { margin: 0; font-size: .88rem; color: var(--text); }
#info-box strong { color: var(--accent); }
#info-box code {
    font-size: .78rem;
    background: rgba(249,115,22,.12);
    color: var(--accent);
    padding: 1px 6px;
    border-radius: 4px;
}

/* ── Tabs ── */
.results-col .tabs .tab-nav button {
    font-size: .82rem !important;
    font-weight: 600;
    border-radius: 8px 8px 0 0 !important;
}
.results-col .tabs .tab-nav button.selected {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* ── Result images ── */
#overlay-img img, #heatmap-img img {
    border-radius: 10px;
    object-fit: contain;
}

/* ── Examples table ── */
.examples-row .gr-examples .gr-sample-btn {
    border-radius: 8px !important;
    font-size: .82rem;
}

/* ── Accordion ── */
.about-section { margin-top: 12px; }
.about-section .label-wrap { font-size: .88rem; }
.about-section table { font-size: .82rem; }
.about-section table td, .about-section table th { padding: 6px 10px; }

/* ── Responsive ── */
@media (max-width: 720px) {
    .app-header h1 { font-size: 1.35rem; }
    #img-upload { min-height: 200px; }
}
"""


# ── Build UI ───────────────────────────────────────────────────────────

with gr.Blocks(
    title="CV Transfer + Grad-CAM",
    css=CSS,
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        secondary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Space Grotesk"),
        radius_size=gr.themes.sizes.radius_lg,
    ),
) as demo:

    # ── Header ──
    gr.HTML("""
    <div class="app-header">
        <h1>🔥 CV Transfer Learning + Grad-CAM</h1>
        <p>Image classification with visual explanations</p>
        <div class="badge-row">
            <span class="badge pytorch">PyTorch</span>
            <span class="badge tf">TensorFlow</span>
            <span class="badge imagenet">ImageNet 1K</span>
            <span class="badge gradio">Grad-CAM</span>
        </div>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── Left: Input panel ──
        with gr.Column(scale=5, min_width=340):
            image_input = gr.Image(
                type="pil",
                label="Upload Image",
                elem_id="img-upload",
                height=280,
                sources=["upload", "clipboard"],
            )

            with gr.Row(elem_classes="settings-row"):
                framework_dd = gr.Dropdown(
                    choices=["PyTorch", "TensorFlow"],
                    value="PyTorch",
                    label="Framework",
                    scale=1,
                    interactive=True,
                )
                model_dd = gr.Dropdown(
                    choices=["ResNet50", "ResNet101", "VGG16", "VGG19",
                             "EfficientNetB0", "MobileNetV2"],
                    value="ResNet50",
                    label="Architecture",
                    scale=1,
                    interactive=True,
                )

            predict_btn = gr.Button(
                "Classify & Visualize",
                variant="primary",
                elem_id="predict-btn",
            )

        # ── Right: Results panel ──
        with gr.Column(scale=6, min_width=360, elem_classes="results-col"):
            info_md = gr.Markdown(
                value="Upload an image and click **Classify & Visualize**",
                elem_id="info-box",
            )

            label_out = gr.HTML(
                value="",
                elem_id="label-out",
            )

            with gr.Tabs():
                with gr.Tab("Grad-CAM Overlay"):
                    overlay_out = gr.Image(
                        label="Overlay",
                        elem_id="overlay-img",
                        height=280,
                        show_label=False,
                    )
                with gr.Tab("Raw Heatmap"):
                    heatmap_out = gr.Image(
                        label="Heatmap",
                        elem_id="heatmap-img",
                        height=280,
                        show_label=False,
                    )

    # ── Examples ──
    example_pairs = [
        ["examples/cat.jpg",   "PyTorch",    "ResNet50"],
        ["examples/dog.jpeg",  "PyTorch",    "MobileNetV2"],
        ["examples/bird.jpg",  "TensorFlow", "VGG16"],
        ["examples/car.jpg",   "PyTorch",    "EfficientNetB0"],
        ["examples/flower.jpg","TensorFlow", "ResNet50"],
    ]
    existing = [e for e in example_pairs if os.path.isfile(e[0])]
    if existing:
        with gr.Row(elem_classes="examples-row"):
            gr.Examples(
                examples=existing,
                inputs=[image_input, framework_dd, model_dd],
                label="Try an example",
            )

    # ── About ──
    with gr.Accordion("About this project", open=False,
                       elem_classes="about-section"):
        gr.Markdown("""
**Grad-CAM** (Gradient-weighted Class Activation Mapping) highlights
the regions that most influenced the model's prediction.

| Model | Params | Speed | Best for |
|---|---|---|---|
| MobileNetV2 | 3.4 M | ⚡ Fast | Quick tests |
| EfficientNetB0 | 5.3 M | ⚡ Fast | Best efficiency |
| ResNet50 | 25.6 M | 🟢 Medium | General purpose |
| ResNet101 | 44.5 M | 🟡 Slower | Higher accuracy |
| VGG16 / 19 | 138 M | 🔴 Slow | Classic arch |

**Tips:** Use clear, well-lit photos · Center the subject ·
Try different models for best results

Built with PyTorch, TensorFlow & Gradio
        """)

    # ── Event wiring ──
    predict_btn.click(
        fn=predict_with_gradcam,
        inputs=[image_input, framework_dd, model_dd],
        outputs=[label_out, overlay_out, heatmap_out, info_md],
        show_progress="minimal",
    )


# ── Launch ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    in_colab = "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ
    in_hf_space = "SPACE_ID" in os.environ

    if in_hf_space:
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    elif in_colab:
        demo.launch(share=True)
    else:
        demo.launch()
