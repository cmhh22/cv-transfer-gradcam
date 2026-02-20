"""
CV Transfer Learning + Grad-CAM Demo
Interactive Gradio app for image classification with visual explanations.
Supports PyTorch & TensorFlow backends.
"""
import os
import time
import numpy as np
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

# ── Model cache (avoid re-downloading weights per request) ─────────────
_MODEL_CACHE = {}


def _get_model(framework: str, model_name: str):
    """Return a cached model instance, lazily created."""
    key = f"{framework}_{model_name}"
    if key not in _MODEL_CACHE:
        if framework == "PyTorch":
            m = PyTorchTransferModel(model_name=model_name.lower(),
                                     num_classes=1000)
        else:
            m = TensorFlowTransferModel(model_name=model_name,
                                        num_classes=1000)
        _MODEL_CACHE[key] = m
    return _MODEL_CACHE[key]


# ── Main prediction function ───────────────────────────────────────────

def predict_with_gradcam(image, framework, model_name):
    """Classify image, return top-5 predictions + Grad-CAM overlays."""
    if image is None:
        return {}, None, None, "⚠️ Please upload an image first."

    try:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        # Framework availability check
        if framework == "PyTorch" and PyTorchTransferModel is None:
            return ({}, None, None,
                    "❌ PyTorch not installed (`pip install torch torchvision`)")
        if framework == "TensorFlow" and TensorFlowTransferModel is None:
            return ({}, None, None,
                    "❌ TensorFlow not installed (`pip install tensorflow`)")

        t0 = time.time()
        model = _get_model(framework, model_name)

        # ── Prediction (top-5) ──
        results = model.predict(image, top_k=5)
        elapsed = time.time() - t0

        # Convert to {label: confidence} for gr.Label
        label_dict = {name: conf for name, conf in results}

        # ── Grad-CAM ──
        fw = "pytorch" if framework == "PyTorch" else "tensorflow"
        target_layer = model.get_target_layer_name()
        gradcam = GradCAM(model.model, framework=fw,
                          target_layer=target_layer)

        heatmap = gradcam.generate_heatmap(image, target_class=None)
        overlay = gradcam.overlay_heatmap(image, heatmap, alpha=0.45)

        # Clean up hooks (PyTorch)
        if fw == "pytorch":
            gradcam.remove_hooks()

        # Build info text
        top_name, top_conf = results[0]
        info = (f"### 🏷️ {top_name}  ({top_conf:.1%})\n"
                f"**Framework:** {framework}  |  "
                f"**Model:** {model_name}  |  "
                f"**Time:** {elapsed:.2f}s  |  "
                f"**Image:** {image.size[0]}×{image.size[1]}")

        return label_dict, overlay, heatmap, info

    except Exception as e:
        return {}, None, None, f"❌ Error: {e}"


# ── Gradio UI ──────────────────────────────────────────────────────────

CSS = """
.main-title { text-align: center; margin-bottom: 0.2em; }
.sub { text-align: center; opacity: 0.7; margin-top: 0; }
footer { display: none !important; }
"""

with gr.Blocks(title="CV Transfer Learning + Grad-CAM",
               css=CSS) as demo:

    gr.HTML("""
    <h1 class='main-title'>🔥 CV Transfer Learning + Grad-CAM</h1>
    <p class='sub'>Image classification with visual explanations &mdash;
        PyTorch &amp; TensorFlow</p>
    """)

    with gr.Row(equal_height=True):
        # ── Left column: inputs ──
        with gr.Column(scale=1, min_width=340):
            image_input = gr.Image(type="pil", label="📷 Upload Image",
                                   height=300)
            with gr.Row():
                framework_dd = gr.Dropdown(
                    choices=["PyTorch", "TensorFlow"],
                    value="PyTorch", label="Framework", scale=1)
                model_dd = gr.Dropdown(
                    choices=["ResNet50", "ResNet101", "VGG16", "VGG19",
                             "EfficientNetB0", "MobileNetV2"],
                    value="ResNet50", label="Architecture", scale=1)

            predict_btn = gr.Button("🔍 Classify & Visualize",
                                    variant="primary", size="lg")

        # ── Right column: results ──
        with gr.Column(scale=1, min_width=340):
            info_md = gr.Markdown(value="*Upload an image and click Classify*")
            label_out = gr.Label(num_top_classes=5,
                                 label="Top-5 Predictions")
            with gr.Tabs():
                with gr.Tab("Grad-CAM Overlay"):
                    overlay_out = gr.Image(label="Overlay", height=300)
                with gr.Tab("Heatmap"):
                    heatmap_out = gr.Image(label="Heatmap", height=300)

    # ── Examples (only if files exist) ──
    example_pairs = [
        ["examples/cat.jpg",   "PyTorch",     "ResNet50"],
        ["examples/dog.jpeg",  "PyTorch",     "MobileNetV2"],
        ["examples/bird.jpg",  "TensorFlow",  "VGG16"],
        ["examples/car.jpg",   "PyTorch",     "EfficientNetB0"],
        ["examples/flower.jpg","TensorFlow",  "ResNet50"],
    ]
    existing = [e for e in example_pairs if os.path.isfile(e[0])]
    if existing:
        gr.Examples(examples=existing,
                    inputs=[image_input, framework_dd, model_dd])

    # ── Info footer ──
    with gr.Accordion("ℹ️  About this project", open=False):
        gr.Markdown("""
**Grad-CAM** (Gradient-weighted Class Activation Mapping) highlights
the regions in an image that most influenced the model's prediction.

| Model | Params | Speed | Best for |
|-------|--------|-------|----------|
| MobileNetV2 | 3.4 M | ⚡ Fast | Mobile / quick tests |
| ResNet50 | 25.6 M | 🟢 Medium | General purpose |
| ResNet101 | 44.5 M | 🟡 Slower | Higher accuracy |
| VGG16 / VGG19 | 138 M | 🔴 Slow | Classic architectures |
| EfficientNetB0 | 5.3 M | ⚡ Fast | Best efficiency |

**Tips for better predictions:**
- Use clear, well-lit photos
- Center the subject in the frame
- Avoid heavy cropping — models work best at 224×224+
- Try different models — EfficientNet & ResNet50 are usually most accurate

Built with ❤️ using PyTorch, TensorFlow & Gradio
        """)

    # ── Connect ──
    predict_btn.click(
        fn=predict_with_gradcam,
        inputs=[image_input, framework_dd, model_dd],
        outputs=[label_out, overlay_out, heatmap_out, info_md],
    )


# ── Launch ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    in_colab = "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ
    demo.launch(share=in_colab)
