"""
CV Transfer Learning + Grad-CAM Demo
Gradio application for image classification with visualization using transfer learning
Supports both PyTorch and TensorFlow backends
"""
import gradio as gr
import numpy as np
from PIL import Image
import os

# Lazy framework imports
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

def predict_with_gradcam(image, framework, model_name):
    """
    Make prediction and generate Grad-CAM heatmap
    
    Args:
        image: Input image
        framework: 'pytorch' or 'tensorflow'
        model_name: Model architecture name
        
    Returns:
        prediction: Class label and confidence
        heatmap: Grad-CAM visualization
    """
    if image is None:
        return "Please upload an image", None, None
    
    try:
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Ensure RGB mode
        image = image.convert('RGB')
        
        if framework == 'PyTorch':
            if PyTorchTransferModel is None:
                return "PyTorch is not installed. Install it with: pip install torch torchvision", None, None
            model = PyTorchTransferModel(model_name=model_name.lower(), num_classes=1000)
            model.load_pretrained()
            prediction, confidence = model.predict(image)
            gradcam = GradCAM(model.model, framework='pytorch')
            heatmap = gradcam.generate_heatmap(image, target_class=None)
            
        else:  # TensorFlow
            if TensorFlowTransferModel is None:
                return "TensorFlow is not installed. Install it with: pip install tensorflow", None, None
            model = TensorFlowTransferModel(model_name=model_name, num_classes=1000)
            model.load_pretrained()
            prediction, confidence = model.predict(image)
            gradcam = GradCAM(model.model, framework='tensorflow')
            heatmap = gradcam.generate_heatmap(image, target_class=None)
        
        result_text = f"**Prediction:** {prediction}\n**Confidence:** {confidence:.2%}"
        
        # Overlay heatmap on original image
        overlay = gradcam.overlay_heatmap(image, heatmap)
        
        return result_text, overlay, heatmap
        
    except Exception as e:
        return f"Error: {str(e)}", None, None

# Create Gradio Interface
with gr.Blocks(title="CV Transfer Learning + Grad-CAM", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔥 CV Transfer Learning + Grad-CAM Visualization
    
    Upload an image to classify it using **Transfer Learning** and visualize the model's decision 
    with **Grad-CAM** (Gradient-weighted Class Activation Mapping).
    
    ### Features:
    - 🔄 **Dual Framework Support**: PyTorch & TensorFlow
    - 🎯 **Multiple Architectures**: ResNet, VGG, EfficientNet, MobileNet
    - 🔍 **Grad-CAM Heatmaps**: See what the model focuses on
    - 🚀 **Pre-trained on ImageNet**: 1000 classes recognition
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image")
            
            framework_dropdown = gr.Dropdown(
                choices=["PyTorch", "TensorFlow"],
                value="PyTorch",
                label="Framework"
            )
            
            model_dropdown = gr.Dropdown(
                choices=["ResNet50", "ResNet101", "VGG16", "VGG19", 
                        "EfficientNetB0", "MobileNetV2"],
                value="ResNet50",
                label="Model Architecture"
            )
            
            predict_btn = gr.Button("🔍 Classify & Visualize", variant="primary")
        
        with gr.Column(scale=1):
            prediction_output = gr.Markdown(label="Prediction Result")
            
            with gr.Tabs():
                with gr.Tab("Grad-CAM Overlay"):
                    overlay_output = gr.Image(label="Grad-CAM Overlay")
                with gr.Tab("Heatmap Only"):
                    heatmap_output = gr.Image(label="Grad-CAM Heatmap")
    
    # Examples — only show entries whose images exist
    example_entries = [
        ["examples/cat.jpg", "PyTorch", "ResNet50"],
        ["examples/dog.jpg", "TensorFlow", "VGG16"],
    ]
    available_examples = [e for e in example_entries if os.path.isfile(e[0])]
    if available_examples:
        gr.Examples(
            examples=available_examples,
            inputs=[image_input, framework_dropdown, model_dropdown],
        )
    
    gr.Markdown("""
    ### About Grad-CAM
    Grad-CAM produces visual explanations for decisions from CNN-based models by highlighting 
    the important regions in the input image that led to the prediction.
    
    ### Models Available
    - **ResNet**: Deep residual networks (50, 101 layers)
    - **VGG**: Very deep networks (16, 19 layers)
    - **EfficientNet**: Efficient scaling
    - **MobileNet**: Lightweight for mobile deployment
    """)
    
    # Connect button to function
    predict_btn.click(
        fn=predict_with_gradcam,
        inputs=[image_input, framework_dropdown, model_dropdown],
        outputs=[prediction_output, overlay_output, heatmap_output]
    )

if __name__ == "__main__":
    # Auto-detect Colab environment and enable share
    import os
    in_colab = 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ
    demo.launch(share=in_colab)
