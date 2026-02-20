"""
TensorFlow Transfer Learning Module
Implements transfer learning with Keras pre-trained models
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import (
    ResNet50, ResNet101, VGG16, VGG19,
    EfficientNetB0, MobileNetV2
)
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List
import json
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# ImageNet labels cache
# ---------------------------------------------------------------------------
_IMAGENET_LABELS: Optional[list] = None


def get_imagenet_labels() -> list:
    """Load ImageNet class names with local file caching."""
    global _IMAGENET_LABELS
    if _IMAGENET_LABELS is not None:
        return _IMAGENET_LABELS

    cache_path = Path(__file__).parent.parent / "data" / "imagenet_labels.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        try:
            _IMAGENET_LABELS = json.loads(cache_path.read_text())
            return _IMAGENET_LABELS
        except Exception:
            pass

    try:
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        with urllib.request.urlopen(url, timeout=10) as resp:
            _IMAGENET_LABELS = json.loads(resp.read().decode())
        cache_path.write_text(json.dumps(_IMAGENET_LABELS))
    except Exception:
        _IMAGENET_LABELS = [f"class_{i}" for i in range(1000)]

    return _IMAGENET_LABELS


# ---------------------------------------------------------------------------
# Preprocessing functions per architecture
# ---------------------------------------------------------------------------
_PREPROCESS_FN = {
    'ResNet50':       keras.applications.resnet.preprocess_input,
    'ResNet101':      keras.applications.resnet.preprocess_input,
    'VGG16':          keras.applications.vgg16.preprocess_input,
    'VGG19':          keras.applications.vgg19.preprocess_input,
    'EfficientNetB0': keras.applications.efficientnet.preprocess_input,
    'MobileNetV2':    keras.applications.mobilenet_v2.preprocess_input,
}

# Best Grad-CAM target layer names per architecture (when using include_top=True)
TARGET_LAYERS = {
    'ResNet50':       'conv5_block3_out',
    'ResNet101':      'conv5_block3_out',
    'VGG16':          'block5_conv3',
    'VGG19':          'block5_conv4',
    'EfficientNetB0': 'top_conv',
    'MobileNetV2':    'out_relu',
}


class TensorFlowTransferModel:
    """Transfer Learning with TensorFlow/Keras pre-trained models."""

    MODEL_MAP = {
        'ResNet50':       ResNet50,
        'ResNet101':      ResNet101,
        'VGG16':          VGG16,
        'VGG19':          VGG19,
        'EfficientNetB0': EfficientNetB0,
        'MobileNetV2':    MobileNetV2,
    }

    def __init__(self, model_name: str = 'ResNet50', num_classes: int = 1000,
                 input_shape: Tuple[int, int, int] = (224, 224, 3)):
        self.model_name = model_name
        self.num_classes = num_classes
        self.input_shape = input_shape

        if self.model_name not in self.MODEL_MAP:
            raise ValueError(
                f"Model '{self.model_name}' not supported. "
                f"Choose from: {list(self.MODEL_MAP.keys())}"
            )

        self.model = self._build_model()
        self.class_names = get_imagenet_labels()

    # ------------------------------------------------------------------ #
    def _build_model(self, pretrained: bool = True) -> keras.Model:
        weights = 'imagenet' if pretrained else None
        model_cls = self.MODEL_MAP[self.model_name]

        if self.num_classes == 1000 and pretrained:
            return model_cls(weights='imagenet', input_shape=self.input_shape)

        base = model_cls(weights=weights, include_top=False,
                         input_shape=self.input_shape)
        return keras.Sequential([
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax'),
        ])

    def load_pretrained(self):
        self.model = self._build_model(pretrained=True)

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Resize, convert and apply model-specific normalization."""
        image = image.convert('RGB').resize(
            (self.input_shape[1], self.input_shape[0]))
        arr = np.array(image, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)

        fn = _PREPROCESS_FN.get(self.model_name)
        if fn is not None:
            arr = fn(arr.copy())  # copy to avoid negative stride issues
        return arr

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict(self, image: Image.Image, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Classify an image, returning top-k results.

        Returns
        -------
        list of (class_name, confidence) tuples sorted by confidence descending.
        """
        arr = self.preprocess_image(image)
        preds = self.model.predict(arr, verbose=0)[0]

        top_indices = np.argsort(preds)[::-1][:top_k]
        results = []
        for idx in top_indices:
            name = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
            results.append((name, round(float(preds[idx]), 5)))
        return results

    def get_target_layer_name(self) -> Optional[str]:
        """Return recommended Grad-CAM target layer for this architecture."""
        return TARGET_LAYERS.get(self.model_name)

    # ------------------------------------------------------------------ #
    # Training helpers
    # ------------------------------------------------------------------ #
    def freeze_layers(self, freeze_base: bool = True):
        if isinstance(self.model, keras.Sequential):
            self.model.layers[0].trainable = not freeze_base
        else:
            self.model.trainable = not freeze_base

    def compile_model(self, learning_rate: float = 0.001,
                      loss: str = 'categorical_crossentropy'):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=['accuracy', 'top_k_categorical_accuracy'],
        )

    def train_model(self, train_ds, val_ds, epochs: int = 10,
                    callbacks: Optional[list] = None):
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.1, patience=3),
            ]
        return self.model.fit(train_ds, validation_data=val_ds,
                              epochs=epochs, callbacks=callbacks)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    def load(self, path: str):
        self.model = keras.models.load_model(path)
