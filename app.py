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
/* ══════════════════════════════════════════════════════════════════
   Design System — Dark + Orange Accent
   Clean, modern, professional.  No gimmicks.
   ══════════════════════════════════════════════════════════════════ */

:root {
    --accent:       #f97316;
    --accent-hover: #ea580c;
    --accent-glow:  rgba(249,115,22,.12);
    --bg:           #0a0a0a;
    --surface:      #141414;
    --surface-2:    #1c1c1c;
    --surface-3:    #262626;
    --border:       rgba(255,255,255,.06);
    --border-hover: rgba(255,255,255,.12);
    --text:         #e5e5e5;
    --text-2:       #a3a3a3;
    --text-3:       #737373;
    --radius:       16px;
    --radius-sm:    10px;
    --radius-xs:    8px;
    --transition:   .2s cubic-bezier(.4,0,.2,1);
}

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }
footer { display: none !important; }
body, .dark, .main, .app, .gradio-container {
    background: var(--bg) !important;
    color: var(--text);
}
.gradio-container {
    max-width: 1120px !important;
    margin: auto;
    padding: 0 16px !important;
}

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 36px 16px 20px;
}
.app-header h1 {
    font-size: 1.65rem;
    font-weight: 800;
    color: #fff;
    margin: 0 0 2px;
    letter-spacing: -.03em;
    line-height: 1.2;
}
.app-header h1 .fire { filter: saturate(1.2); }
.app-header .subtitle {
    font-size: .88rem;
    color: var(--text-3);
    margin: 0;
    font-weight: 400;
}
.app-header .pill-row {
    display: flex;
    justify-content: center;
    gap: 6px;
    margin-top: 14px;
    flex-wrap: wrap;
}
.pill {
    font-size: .68rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 100px;
    letter-spacing: .02em;
    border: 1px solid transparent;
    backdrop-filter: blur(4px);
}
.pill.pt  { background: rgba(238,76,44,.08); color: #ff7b6b; border-color: rgba(238,76,44,.18); }
.pill.tf  { background: rgba(255,111,0,.08); color: #ffba70; border-color: rgba(255,111,0,.18); }
.pill.gc  { background: rgba(249,115,22,.08); color: #f97316; border-color: rgba(249,115,22,.18); }
.pill.in  { background: rgba(251,191,36,.06); color: #fbbf24; border-color: rgba(251,191,36,.15); }

/* ── Glass card wrapper ── */
.panel-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0;
    overflow: hidden;
}

/* ── Predict button ── */
#predict-btn {
    width: 100% !important;
    height: 46px !important;
    min-height: 46px !important;
    max-height: 46px !important;
    font-size: .9rem !important;
    font-weight: 700 !important;
    letter-spacing: .02em;
    border-radius: var(--radius-sm) !important;
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    cursor: pointer;
    transition: all var(--transition);
    box-shadow: 0 0 0 0 transparent, 0 2px 8px rgba(249,115,22,.25);
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: hidden !important;
    padding: 0 20px !important;
    flex-shrink: 0 !important;
    position: relative;
}
#predict-btn::before {
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(180deg, rgba(255,255,255,.12) 0%, transparent 60%);
    pointer-events: none;
    border-radius: inherit;
}
#predict-btn:hover {
    background: var(--accent-hover) !important;
    box-shadow: 0 0 20px rgba(249,115,22,.2), 0 4px 12px rgba(249,115,22,.3);
    transform: translateY(-1px);
}
#predict-btn:active {
    transform: translateY(0) scale(.99);
    box-shadow: 0 0 0 0 transparent, 0 1px 4px rgba(249,115,22,.2);
}

/* ── Kill extra spinners ── */
#predict-btn .wrap, #predict-btn .loading,
.results-col .progress-bar, .results-col .wrap.default,
.results-col > div > .wrap.default,
#overlay-img .wrap, #heatmap-img .wrap, #info-box .wrap {
    display: none !important;
}

/* ── Predictions panel ── */
#label-out {
    position: relative;
    min-height: 80px;
    background: var(--surface);
    border-radius: var(--radius);
    padding: 14px 18px;
    border: 1px solid var(--border);
}

/* ── Dropdowns ── */
.settings-row select, .settings-row .gr-dropdown,
.settings-row input {
    border-radius: var(--radius-xs) !important;
    background: var(--surface-2) !important;
    border-color: var(--border) !important;
    transition: border-color var(--transition);
}
.settings-row select:focus, .settings-row .gr-dropdown:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow);
}

/* ── Image upload ── */
#img-upload {
    border: 2px dashed rgba(255,255,255,.08) !important;
    border-radius: var(--radius) !important;
    transition: border-color var(--transition), background var(--transition);
    min-height: 270px;
    background: var(--surface) !important;
}
#img-upload:hover {
    border-color: rgba(249,115,22,.35) !important;
    background: rgba(249,115,22,.02) !important;
}

/* ── Info box ── */
#info-box {
    min-height: 44px;
    padding: 12px 16px;
    background: var(--surface-2);
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
}
#info-box p { margin: 0; font-size: .86rem; color: var(--text); line-height: 1.5; }
#info-box strong { color: #fff; }
#info-box code {
    font-size: .75rem;
    background: var(--accent-glow);
    color: var(--accent);
    padding: 2px 8px;
    border-radius: 5px;
    font-weight: 500;
}

/* ── Tabs ── */
.results-col .tabs .tab-nav {
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.results-col .tabs .tab-nav button {
    font-size: .8rem !important;
    font-weight: 600;
    padding: 10px 18px !important;
    border-radius: var(--radius-xs) var(--radius-xs) 0 0 !important;
    color: var(--text-3) !important;
    border: none !important;
    background: transparent !important;
    transition: color var(--transition), background var(--transition);
    position: relative;
}
.results-col .tabs .tab-nav button:hover {
    color: var(--text-2) !important;
    background: rgba(255,255,255,.03) !important;
}
.results-col .tabs .tab-nav button.selected {
    color: var(--accent) !important;
    background: rgba(249,115,22,.05) !important;
}
.results-col .tabs .tab-nav button.selected::after {
    content: '';
    position: absolute;
    bottom: -1px; left: 12px; right: 12px;
    height: 2px;
    background: var(--accent);
    border-radius: 2px 2px 0 0;
}

/* ── Result images ── */
#overlay-img img, #heatmap-img img {
    border-radius: var(--radius-sm);
    object-fit: contain;
}

/* ── Examples ── */
.examples-row {
    margin-top: 8px;
}
.examples-row .gr-examples {
    background: transparent !important;
    border: none !important;
}
.examples-row .gr-examples .gr-sample-btn {
    border-radius: var(--radius-xs) !important;
    font-size: .8rem;
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    transition: all var(--transition);
}
.examples-row .gr-examples .gr-sample-btn:hover {
    border-color: rgba(249,115,22,.3) !important;
    background: rgba(249,115,22,.04) !important;
}

/* ── Accordion ── */
.about-section { margin-top: 12px; }
.about-section .label-wrap {
    font-size: .84rem;
    color: var(--text-3);
    border-radius: var(--radius-sm);
}
.about-section table {
    font-size: .8rem;
    border-collapse: separate;
    border-spacing: 0;
}
.about-section table th {
    padding: 8px 12px;
    color: var(--text-3);
    font-weight: 600;
    font-size: .72rem;
    text-transform: uppercase;
    letter-spacing: .04em;
    border-bottom: 1px solid var(--border);
}
.about-section table td {
    padding: 7px 12px;
    border-bottom: 1px solid rgba(255,255,255,.03);
}

/* ── Block labels (Gradio internal) ── */
.gr-block-label, .gr-input-label, label.svelte-1b6s6s {
    font-size: .75rem !important;
    font-weight: 600 !important;
    color: var(--text-3) !important;
    text-transform: uppercase;
    letter-spacing: .05em;
}

/* ── Responsive ── */
@media (max-width: 720px) {
    .app-header h1 { font-size: 1.3rem; }
    .app-header { padding: 24px 12px 14px; }
    #img-upload { min-height: 200px; }
    .pill { font-size: .62rem; padding: 3px 8px; }
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
        font=gr.themes.GoogleFont("Inter"),
        radius_size=gr.themes.sizes.radius_lg,
    ),
) as demo:

    # ── Header ──
    gr.HTML("""
    <div class="app-header">
        <h1><span class="fire">🔥</span> CV Transfer Learning + Grad-CAM</h1>
        <p class="subtitle">Image classification with visual explanations</p>
        <div class="pill-row">
            <span class="pill pt">PyTorch</span>
            <span class="pill tf">TensorFlow</span>
            <span class="pill in">ImageNet 1K</span>
            <span class="pill gc">Grad-CAM</span>
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
