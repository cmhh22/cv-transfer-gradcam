---
title: CV Transfer Learning + Grad-CAM
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
license: mit
python_version: "3.11"
---

# 🔥 CV Transfer Learning + Grad-CAM

Transfer Learning with Grad-CAM visualization in both **PyTorch** and **TensorFlow**.

## Features

- 🔄 **Dual Framework Support**: Switch between PyTorch and TensorFlow
- 🏗️ **Multiple Architectures**: ResNet, VGG, EfficientNet, MobileNet
- 🔍 **Grad-CAM Visualization**: Understand model decisions with heatmaps
- 🎨 **Interactive Demo**: Easy-to-use Gradio interface
- 🚀 **Pre-trained Models**: Immediate results with ImageNet models

## How to Use

1. Upload an image 📸
2. Select a framework (PyTorch or TensorFlow) ⚙️
3. Choose a model architecture 🏗️
4. Click "Classify & Visualize" 🔍
5. View predictions and Grad-CAM heatmaps 🖼️

## Models Available

- **ResNet50/101**: Deep residual networks
- **VGG16/19**: Very deep convolutional networks
- **EfficientNetB0**: Efficient compound scaling
- **MobileNetV2**: Lightweight mobile architecture

## About Grad-CAM

Grad-CAM produces visual explanations for CNN decisions by highlighting important regions in the input image that influenced the prediction.

## Repository

Full source code and documentation: [GitHub](https://github.com/cmhh22/cv-transfer-gradcam)

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{cv-transfer-gradcam,
  author = {Your Name},
  title = {CV Transfer Learning + Grad-CAM},
  year = {2026},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/spaces/cmhh22/cv-transfer-gradcam}}
}
```

## License

MIT License - see LICENSE file for details
