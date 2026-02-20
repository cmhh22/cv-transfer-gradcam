# 🔥 CV Transfer Learning + Grad-CAM

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-yellow.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Transfer Learning implementation with Grad-CAM visualization in both **PyTorch** and **TensorFlow**. 
Features an interactive **Gradio** demo for image classification with visual explanations.

## 🎯 Features

- **🔄 Dual Framework Support**: Seamlessly switch between PyTorch and TensorFlow
- **🏗️ Multiple Architectures**: ResNet, VGG, EfficientNet, MobileNet
- **🔍 Grad-CAM Visualization**: Understand model decisions with heatmaps
- **🚀 Transfer Learning**: Fine-tune on custom datasets
- **🎨 Interactive Demo**: Gradio web interface
- **☁️ HuggingFace Spaces**: Ready for deployment

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Models](#models)
- [Grad-CAM](#grad-cam)
- [Training](#training)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [License](#license)

## 🚀 Installation

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (optional, for faster training)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/cv-transfer-gradcam.git
cd cv-transfer-gradcam

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🎬 Quick Start

### Run Gradio Demo

```bash
python app.py
```

The application will launch at `http://localhost:7860`

### Python API

```python
from PIL import Image
from src.pytorch_transfer import PyTorchTransferModel
from src.gradcam import GradCAM

# Load model
model = PyTorchTransferModel(model_name='resnet50', num_classes=1000)
model.load_pretrained()

# Make prediction
image = Image.open('path/to/image.jpg')
prediction, confidence = model.predict(image)
print(f"Prediction: {prediction} ({confidence:.2%})")

# Generate Grad-CAM
gradcam = GradCAM(model.model, framework='pytorch')
heatmap = gradcam.generate_heatmap(image)
overlay = gradcam.overlay_heatmap(image, heatmap)
overlay.save('gradcam_result.jpg')
```

## 📖 Usage

### PyTorch Example

```python
from src.pytorch_transfer import PyTorchTransferModel

# Initialize model
model = PyTorchTransferModel(
    model_name='resnet50',
    num_classes=10,  # Your custom number of classes
    pretrained=True
)

# Fine-tune on your dataset
model.train_model(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    lr=0.001
)

# Save model
model.save('models/my_model.pth')
```

### TensorFlow Example

```python
from src.tensorflow_transfer import TensorFlowTransferModel

# Initialize model
model = TensorFlowTransferModel(
    model_name='ResNet50',
    num_classes=10,
    input_shape=(224, 224, 3)
)

# Compile and train
model.compile_model(learning_rate=0.001)
model.train_model(
    train_ds=train_dataset,
    val_ds=val_dataset,
    epochs=10
)

# Save model
model.save('models/my_model.h5')
```

## 🏗️ Models

### Supported Architectures

| Model | PyTorch | TensorFlow | Parameters | Input Size |
|-------|---------|------------|------------|------------|
| ResNet50 | ✅ | ✅ | 25.6M | 224x224 |
| ResNet101 | ✅ | ✅ | 44.5M | 224x224 |
| VGG16 | ✅ | ✅ | 138M | 224x224 |
| VGG19 | ✅ | ✅ | 144M | 224x224 |
| EfficientNetB0 | ✅ | ✅ | 5.3M | 224x224 |
| MobileNetV2 | ✅ | ✅ | 3.5M | 224x224 |

## 🔍 Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) produces visual explanations for CNN decisions.

### How it works:

1. **Forward Pass**: Image flows through the network
2. **Backward Pass**: Gradients computed for target class
3. **Weighted Combination**: Activations weighted by gradients
4. **Heatmap**: ReLU applied and upsampled to input size

### Generate Grad-CAM

```python
from src.gradcam import GradCAM

# PyTorch
gradcam = GradCAM(model.model, framework='pytorch', target_layer='layer4')
heatmap = gradcam.generate_heatmap(image, target_class=285)  # 285 = cat

# TensorFlow
gradcam = GradCAM(model.model, framework='tensorflow', target_layer='conv5_block3_out')
heatmap = gradcam.generate_heatmap(image, target_class=285)
```

## 🎓 Training

### Prepare Dataset

```python
from src.utils import prepare_dataset

# Organize your data:
# data/
#   train/
#     class1/
#     class2/
#   val/
#     class1/
#     class2/

train_loader, val_loader = prepare_dataset(
    data_dir='data',
    batch_size=32,
    img_size=224
)
```

### Training Script

```bash
# PyTorch
python src/train_pytorch.py --model resnet50 --epochs 20 --lr 0.001 --data data/

# TensorFlow
python src/train_tensorflow.py --model ResNet50 --epochs 20 --lr 0.001 --data data/
```

## ☁️ Deployment

### HuggingFace Spaces

1. Create a new Space on [HuggingFace](https://huggingface.co/spaces)
2. Choose **Gradio** as SDK
3. Push your code:

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE
git push hf main
```

### Docker

```bash
# Build image
docker build -t cv-transfer-gradcam .

# Run container
docker run -p 7860:7860 cv-transfer-gradcam
```

## 📁 Project Structure

```
cv-transfer-gradcam/
├── app.py                      # Gradio application
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore rules
├── Dockerfile                  # Docker configuration
├── src/
│   ├── __init__.py
│   ├── pytorch_transfer.py    # PyTorch transfer learning
│   ├── tensorflow_transfer.py # TensorFlow transfer learning
│   ├── gradcam.py             # Grad-CAM implementation
│   ├── utils.py               # Utility functions
│   ├── train_pytorch.py       # PyTorch training script
│   └── train_tensorflow.py    # TensorFlow training script
├── models/                     # Saved models
├── data/                       # Dataset directory
│   ├── train/
│   └── val/
├── examples/                   # Example images
├── notebooks/                  # Jupyter notebooks
│   ├── 01_pytorch_tutorial.ipynb
│   ├── 02_tensorflow_tutorial.ipynb
│   └── 03_gradcam_analysis.ipynb
└── docs/                       # Additional documentation
    ├── TRAINING.md
    └── API.md
```

## 📊 Examples

See the `notebooks/` directory for detailed tutorials:

- **PyTorch Tutorial**: Transfer learning from scratch
- **TensorFlow Tutorial**: Fine-tuning best practices
- **Grad-CAM Analysis**: Understanding model decisions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Team** for the excellent deep learning framework
- **TensorFlow Team** for Keras and TensorFlow
- **Gradio Team** for the intuitive UI framework
- **Selvaraju et al.** for the Grad-CAM paper

## 📚 References

- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)

## 📧 Contact

For questions or feedback, please open an issue or contact [your-email@example.com](mailto:your-email@example.com)

---

⭐ If you found this project helpful, please give it a star!
