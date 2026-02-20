# Quick Start Guide

## Installation

```bash
# Clone repository
git clone https://github.com/cmhh22/cv-transfer-gradcam.git
cd cv-transfer-gradcam

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Run Gradio Demo

```bash
python app.py
```

Open http://localhost:7860 in your browser.

## Basic Usage

### PyTorch

```python
from PIL import Image
from src.pytorch_transfer import PyTorchTransferModel
from src.gradcam import GradCAM

# Load model
model = PyTorchTransferModel(model_name='resnet50')
model.load_pretrained()

# Predict
image = Image.open('image.jpg')
results = model.predict(image, top_k=5)
for name, conf in results:
    print(f"{name}: {conf:.2%}")

# Grad-CAM
target_layer = model.get_target_layer_name()
gradcam = GradCAM(model.model, framework='pytorch', target_layer=target_layer)
heatmap = gradcam.generate_heatmap(image)
overlay = gradcam.overlay_heatmap(image, heatmap)
overlay.save('result.jpg')
```

### TensorFlow

```python
from src.tensorflow_transfer import TensorFlowTransferModel
from src.gradcam import GradCAM

# Load model
model = TensorFlowTransferModel(model_name='ResNet50')

# Predict
results = model.predict(image, top_k=5)
for name, conf in results:
    print(f"{name}: {conf:.2%}")

# Grad-CAM
target_layer = model.get_target_layer_name()
gradcam = GradCAM(model.model, framework='tensorflow', target_layer=target_layer)
heatmap = gradcam.generate_heatmap(image)
```

## Training Custom Model

```python
from src.utils import prepare_pytorch_dataset

# Prepare data
train_loader, val_loader = prepare_pytorch_dataset('data/')

# Train
model = PyTorchTransferModel('resnet50', num_classes=10)
model.freeze_layers()
model.train_model(train_loader, val_loader, epochs=20)
```

## Deploy to HuggingFace Spaces

1. Create account on [HuggingFace](https://huggingface.co)
2. Create new Space with Gradio SDK
3. Push code:

```bash
git remote add hf https://huggingface.co/spaces/cmhh22/cv-transfer-gradcam
git push hf main
```

## Docker

```bash
docker build -t cv-transfer-gradcam .
docker run -p 7860:7860 cv-transfer-gradcam
```

## Next Steps

- Check `notebooks/` for detailed tutorials
- Read `docs/TRAINING.md` for training guide
- See `docs/API.md` for complete API reference
