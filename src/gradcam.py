"""
Grad-CAM Implementation for both PyTorch and TensorFlow
Gradient-weighted Class Activation Mapping
"""
import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple

# Lazy imports — frameworks loaded only when needed
try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

class GradCAM:
    """Grad-CAM visualization for CNN models"""
    
    def __init__(self, model, framework: str = 'pytorch', target_layer: Optional[str] = None):
        """
        Initialize Grad-CAM
        
        Args:
            model: The CNN model
            framework: 'pytorch' or 'tensorflow'
            target_layer: Target layer for visualization (auto-detected if None)
        """
        self.model = model
        self.framework = framework.lower()
        self.target_layer = target_layer
        
        # Storage for activations and gradients
        self.activations = None
        self.gradients = None
        
        # Set up hooks/tape
        if self.framework == 'pytorch':
            self._setup_pytorch_hooks()
        elif self.framework == 'tensorflow':
            pass  # TensorFlow uses GradientTape dynamically
        else:
            raise ValueError(f"Framework {framework} not supported")
    
    def _setup_pytorch_hooks(self):
        """Setup forward and backward hooks for PyTorch"""
        # Auto-detect target layer if not specified
        if self.target_layer is None:
            # Get the last convolutional layer
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    self.target_layer_module = module
        else:
            self.target_layer_module = dict(self.model.named_modules())[self.target_layer]
        
        # Forward hook to capture activations
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        # Backward hook to capture gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer_module.register_forward_hook(forward_hook)
        self.target_layer_module.register_full_backward_hook(backward_hook)
    
    def generate_heatmap(self, image: Image.Image, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap
        
        Args:
            image: Input image (PIL Image)
            target_class: Target class index (uses predicted class if None)
            
        Returns:
            Heatmap as numpy array
        """
        if self.framework == 'pytorch':
            return self._generate_heatmap_pytorch(image, target_class)
        else:
            return self._generate_heatmap_tensorflow(image, target_class)
    
    def _generate_heatmap_pytorch(self, image: Image.Image, target_class: Optional[int] = None) -> np.ndarray:
        """Generate Grad-CAM heatmap using PyTorch"""
        from torchvision import transforms
        
        # Preprocess image
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = preprocess(image).unsqueeze(0)
        
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            self.model = self.model.cuda()
        
        input_tensor.requires_grad = True
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Get gradients and activations
        gradients = self.gradients[0].cpu().numpy()  # [C, H, W]
        activations = self.activations[0].cpu().numpy()  # [C, H, W]
        
        # Calculate weights (global average pooling of gradients)
        weights = np.mean(gradients, axis=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # [H, W]
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize to input image size
        cam = cv2.resize(cam, (224, 224))
        
        return cam
    
    def _generate_heatmap_tensorflow(self, image: Image.Image, target_class: Optional[int] = None) -> np.ndarray:
        """Generate Grad-CAM heatmap using TensorFlow"""
        from tensorflow import keras
        
        # Preprocess image
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Determine preprocessing based on model name
        model_name = getattr(self.model, 'name', '').lower()
        if 'vgg' in model_name:
            img_array = keras.applications.vgg16.preprocess_input(img_array)
        elif 'efficientnet' in model_name:
            img_array = keras.applications.efficientnet.preprocess_input(img_array)
        elif 'mobilenet' in model_name:
            img_array = keras.applications.mobilenet_v2.preprocess_input(img_array)
        else:
            # Default to ResNet preprocessing
            img_array = keras.applications.resnet.preprocess_input(img_array)
        
        # Auto-detect last convolutional layer if not specified
        if self.target_layer is None:
            last_conv_layer = None
            for layer in reversed(self.model.layers):
                # Check by layer type name (most reliable)
                layer_class = layer.__class__.__name__
                if 'Conv2D' in layer_class or 'Conv2d' in layer_class:
                    last_conv_layer = layer
                    break
                # Fallback: check output shape
                try:
                    output_shape = layer.output_shape
                    if isinstance(output_shape, list):
                        output_shape = output_shape[0]
                    if isinstance(output_shape, tuple) and len(output_shape) == 4:
                        last_conv_layer = layer
                        break
                except (AttributeError, RuntimeError, ValueError):
                    continue
            if last_conv_layer is None:
                raise ValueError("Could not auto-detect a convolutional layer in the model")
        else:
            last_conv_layer = self.model.get_layer(self.target_layer)
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[last_conv_layer.output, self.model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            
            if target_class is None:
                target_class = tf.argmax(predictions[0])
            
            class_channel = predictions[:, target_class]
        
        # Gradient of the predicted class with respect to output feature map
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by the pooled gradients
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Average of weighted feature maps
        heatmap = np.mean(conv_outputs, axis=-1)
        
        # Apply ReLU
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Resize to input image size
        heatmap = cv2.resize(heatmap, (224, 224))
        
        return heatmap
    
    def overlay_heatmap(self, image: Image.Image, heatmap: np.ndarray, 
                       alpha: float = 0.4, colormap: int = cv2.COLORMAP_JET) -> Image.Image:
        """
        Overlay heatmap on original image
        
        Args:
            image: Original image
            heatmap: Grad-CAM heatmap
            alpha: Transparency of overlay (0-1)
            colormap: OpenCV colormap
            
        Returns:
            Overlayed image as PIL Image
        """
        # Resize image to match heatmap
        img = image.resize((224, 224))
        img_array = np.array(img)
        
        # Convert heatmap to RGB
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlayed = cv2.addWeighted(img_array, 1 - alpha, heatmap_colored, alpha, 0)
        
        return Image.fromarray(overlayed)
    
    def save_visualization(self, image: Image.Image, heatmap: np.ndarray, 
                          output_path: str, alpha: float = 0.4):
        """Save Grad-CAM visualization"""
        overlayed = self.overlay_heatmap(image, heatmap, alpha)
        overlayed.save(output_path)
        print(f"Visualization saved to {output_path}")
    
    @staticmethod
    def visualize_multiple_classes(image: Image.Image, model, classes: list, 
                                   framework: str = 'pytorch') -> dict:
        """
        Generate Grad-CAM for multiple target classes
        
        Args:
            image: Input image
            model: CNN model
            classes: List of class indices
            framework: 'pytorch' or 'tensorflow'
            
        Returns:
            Dictionary mapping class indices to heatmaps
        """
        gradcam = GradCAM(model, framework=framework)
        results = {}
        
        for cls in classes:
            heatmap = gradcam.generate_heatmap(image, target_class=cls)
            results[cls] = heatmap
        
        return results
