"""
Grad-CAM Implementation for both PyTorch and TensorFlow
Gradient-weighted Class Activation Mapping

Properly hooks into known target layers per architecture for reliable
heatmap generation.
"""
import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple

# Lazy imports
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

try:
    import tensorflow as tf
except ImportError:
    tf = None


class GradCAM:
    """Grad-CAM visualization for CNN models."""

    def __init__(self, model, framework: str = 'pytorch',
                 target_layer: Optional[str] = None):
        """
        Parameters
        ----------
        model : torch.nn.Module or tf.keras.Model
        framework : 'pytorch' or 'tensorflow'
        target_layer : layer name for Grad-CAM.
            If None the class will try to auto-detect the best one.
        """
        self.model = model
        self.framework = framework.lower()
        self.target_layer = target_layer

        self._activations = None
        self._gradients = None
        self._hooks = []  # track hooks for cleanup

        if self.framework == 'pytorch':
            self._setup_pytorch_hooks()
        elif self.framework != 'tensorflow':
            raise ValueError(f"Framework '{framework}' not supported")

    # ------------------------------------------------------------------
    # PyTorch hooks
    # ------------------------------------------------------------------
    def _find_pytorch_target(self):
        """Find the target conv layer by name or auto-detect."""
        modules = dict(self.model.named_modules())

        if self.target_layer and self.target_layer in modules:
            return modules[self.target_layer]

        # Auto-detect by known architecture patterns
        name = type(self.model).__name__.lower()

        # ResNet → layer4 (last bottleneck block)
        if 'resnet' in name and 'layer4' in modules:
            return modules['layer4']

        # VGG → last Conv2d in features
        if 'vgg' in name:
            for child in reversed(list(self.model.features.children())):
                if isinstance(child, torch.nn.Conv2d):
                    return child

        # EfficientNet / MobileNet → last Conv2d anywhere
        last_conv = None
        for _, m in self.model.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                last_conv = m
        if last_conv is not None:
            return last_conv

        raise RuntimeError("Could not auto-detect a convolutional layer")

    def _setup_pytorch_hooks(self):
        target = self._find_pytorch_target()

        def fwd(module, inp, out):
            self._activations = out.detach()

        def bwd(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._hooks.append(target.register_forward_hook(fwd))
        self._hooks.append(target.register_full_backward_hook(bwd))

    def remove_hooks(self):
        """Remove all registered hooks (call when done)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_heatmap(self, image: Image.Image,
                         target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for *image*.

        Returns an (H, W) float32 array normalised to [0, 1] at the
        original image resolution.
        """
        if self.framework == 'pytorch':
            return self._heatmap_pytorch(image, target_class)
        return self._heatmap_tensorflow(image, target_class)

    # ------------------------------------------------------------------
    # PyTorch
    # ------------------------------------------------------------------
    def _heatmap_pytorch(self, image: Image.Image,
                         target_class: Optional[int] = None) -> np.ndarray:
        from torchvision import transforms

        orig_w, orig_h = image.size

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        inp = preprocess(image.convert('RGB')).unsqueeze(0)
        device = next(self.model.parameters()).device
        inp = inp.to(device).requires_grad_(True)

        self.model.eval()
        out = self.model(inp)

        if target_class is None:
            target_class = out.argmax(dim=1).item()

        self.model.zero_grad()
        out[0, target_class].backward()

        grads = self._gradients[0].cpu().numpy()       # (C, h, w)
        acts  = self._activations[0].cpu().numpy()     # (C, h, w)

        weights = grads.mean(axis=(1, 2))              # GAP → (C,)
        cam = np.einsum('c,chw->hw', weights, acts)    # weighted sum
        cam = np.maximum(cam, 0)                       # ReLU

        if cam.max() > 0:
            cam /= cam.max()

        # Resize to original image dimensions
        cam = cv2.resize(cam, (orig_w, orig_h))
        return cam

    # ------------------------------------------------------------------
    # TensorFlow
    # ------------------------------------------------------------------
    def _find_tf_target_layer(self):
        """Find the target conv layer in a Keras model."""
        if self.target_layer:
            return self.model.get_layer(self.target_layer)

        # Try known layer names first
        known_names = [
            'conv5_block3_out',  # ResNet50/101
            'block5_conv3',      # VGG16
            'block5_conv4',      # VGG19
            'top_conv',          # EfficientNet
            'out_relu',          # MobileNetV2
            'Conv_1',            # MobileNetV2 alt
        ]
        for name in known_names:
            try:
                return self.model.get_layer(name)
            except ValueError:
                continue

        # Fallback: last layer whose output is 4-D
        for layer in reversed(self.model.layers):
            cls_name = layer.__class__.__name__
            if 'Conv2D' in cls_name or 'Conv2d' in cls_name:
                return layer
            try:
                shape = layer.output_shape
                if isinstance(shape, list):
                    shape = shape[0]
                if isinstance(shape, tuple) and len(shape) == 4:
                    return layer
            except Exception:
                continue

        raise RuntimeError("Could not auto-detect a conv layer in the TF model")

    def _heatmap_tensorflow(self, image: Image.Image,
                            target_class: Optional[int] = None) -> np.ndarray:
        from tensorflow import keras

        orig_w, orig_h = image.size

        img = image.convert('RGB').resize((224, 224))
        arr = np.array(img, dtype=np.float32)
        arr = np.expand_dims(arr, 0)

        # Model-specific preprocessing
        model_name = getattr(self.model, 'name', '').lower()
        if 'vgg' in model_name:
            arr = keras.applications.vgg16.preprocess_input(arr.copy())
        elif 'efficientnet' in model_name:
            arr = keras.applications.efficientnet.preprocess_input(arr.copy())
        elif 'mobilenet' in model_name:
            arr = keras.applications.mobilenet_v2.preprocess_input(arr.copy())
        else:
            arr = keras.applications.resnet.preprocess_input(arr.copy())

        conv_layer = self._find_tf_target_layer()

        grad_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[conv_layer.output, self.model.output],
        )

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(arr)
            if target_class is None:
                target_class = tf.argmax(preds[0])
            loss = preds[:, target_class]

        grads = tape.gradient(loss, conv_out)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        conv_out = conv_out[0].numpy()

        cam = np.einsum('hwc,c->hw', conv_out, weights)
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam /= cam.max()

        cam = cv2.resize(cam, (orig_w, orig_h))
        return cam

    # ------------------------------------------------------------------
    # Overlay
    # ------------------------------------------------------------------
    def overlay_heatmap(self, image: Image.Image, heatmap: np.ndarray,
                        alpha: float = 0.4,
                        colormap: int = cv2.COLORMAP_JET) -> Image.Image:
        """Overlay heatmap on original image (preserving its resolution)."""
        img_arr = np.array(image.convert('RGB'))
        h, w = img_arr.shape[:2]

        hm = cv2.resize(heatmap, (w, h))
        hm_color = cv2.applyColorMap(np.uint8(255 * hm), colormap)
        hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

        blended = cv2.addWeighted(img_arr, 1 - alpha, hm_color, alpha, 0)
        return Image.fromarray(blended)

    def save_visualization(self, image: Image.Image, heatmap: np.ndarray,
                           path: str, alpha: float = 0.4):
        self.overlay_heatmap(image, heatmap, alpha).save(path)
