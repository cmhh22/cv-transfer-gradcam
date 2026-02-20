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
from typing import Tuple, Optional

class TensorFlowTransferModel:
    """Transfer Learning with TensorFlow/Keras pre-trained models"""
    
    def __init__(self, model_name: str = 'ResNet50', num_classes: int = 1000, 
                 input_shape: Tuple[int, int, int] = (224, 224, 3)):
        """
        Initialize transfer learning model
        
        Args:
            model_name: Name of the architecture ('ResNet50', 'VGG16', etc.)
            num_classes: Number of output classes
            input_shape: Input image shape (height, width, channels)
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.input_shape = input_shape
        
        # Load model
        self.model = self._build_model()
        
        # ImageNet classes
        self.class_names = self._load_imagenet_classes()
    
    def _build_model(self, pretrained: bool = True) -> keras.Model:
        """Build the transfer learning model"""
        weights = 'imagenet' if pretrained else None

        model_map = {
            'ResNet50': ResNet50,
            'ResNet101': ResNet101,
            'VGG16': VGG16,
            'VGG19': VGG19,
            'EfficientNetB0': EfficientNetB0,
            'MobileNetV2': MobileNetV2,
        }

        if self.model_name not in model_map:
            raise ValueError(f"Model {self.model_name} not supported. Choose from: {list(model_map.keys())}")

        model_cls = model_map[self.model_name]

        if self.num_classes == 1000 and pretrained:
            # Use full pretrained model with top (ImageNet classification)
            model = model_cls(weights='imagenet', input_shape=self.input_shape)
        else:
            # Build custom classification head
            base_model = model_cls(weights=weights, include_top=False, input_shape=self.input_shape)
            model = keras.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation='softmax')
            ])

        return model
    
    def _load_imagenet_classes(self) -> list:
        """Load ImageNet class names"""
        try:
            import urllib.request
            import json
            
            url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
            with urllib.request.urlopen(url) as response:
                classes = json.loads(response.read().decode())
            return classes
        except:
            return [f"class_{i}" for i in range(1000)]
    
    def load_pretrained(self):
        """Load pretrained ImageNet weights"""
        self.model = self._build_model(pretrained=True)
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for model input"""
        # Resize image
        image = image.resize((self.input_shape[1], self.input_shape[0]))
        
        # Convert to array
        img_array = np.array(image, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess based on model
        if self.model_name.startswith('ResNet'):
            img_array = keras.applications.resnet.preprocess_input(img_array)
        elif self.model_name.startswith('VGG'):
            img_array = keras.applications.vgg16.preprocess_input(img_array)
        elif self.model_name.startswith('EfficientNet'):
            img_array = keras.applications.efficientnet.preprocess_input(img_array)
        elif self.model_name.startswith('MobileNet'):
            img_array = keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        return img_array
    
    def predict(self, image: Image.Image, top_k: int = 1) -> Tuple[str, float]:
        """
        Make prediction on a single image
        
        Args:
            image: PIL Image
            top_k: Return top k predictions
            
        Returns:
            Tuple of (prediction, confidence)
        """
        # Preprocess image
        img_array = self.preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        
        # Get top predictions
        top_indices = np.argsort(predictions[0])[::-1][:top_k]
        
        if top_k == 1:
            pred_class = top_indices[0]
            pred_prob = predictions[0][pred_class]
            pred_name = self.class_names[pred_class] if pred_class < len(self.class_names) else f"class_{pred_class}"
            return pred_name, float(pred_prob)
        else:
            results = []
            for idx in top_indices:
                pred_prob = predictions[0][idx]
                pred_name = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
                results.append((pred_name, float(pred_prob)))
            return results
    
    def freeze_layers(self, freeze_base: bool = True):
        """Freeze base model layers for fine-tuning"""
        if isinstance(self.model, keras.Sequential):
            base_model = self.model.layers[0]
            base_model.trainable = not freeze_base
        else:
            self.model.trainable = not freeze_base
    
    def compile_model(self, learning_rate: float = 0.001, loss: str = 'categorical_crossentropy'):
        """Compile the model"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
    
    def train_model(self, train_ds, val_ds, epochs: int = 10, 
                   callbacks: Optional[list] = None):
        """
        Train the model
        
        Args:
            train_ds: Training dataset
            val_ds: Validation dataset
            epochs: Number of training epochs
            callbacks: List of Keras callbacks
        """
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.1,
                    patience=3
                )
            ]
        
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history
    
    def save(self, path: str):
        """Save model"""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")
    
    def get_feature_extractor(self, layer_name: Optional[str] = None):
        """Get intermediate features from a specific layer"""
        if layer_name is None:
            # Return features before final layer
            if isinstance(self.model, keras.Sequential):
                return keras.Model(
                    inputs=self.model.input,
                    outputs=self.model.layers[-2].output
                )
            else:
                return keras.Model(
                    inputs=self.model.input,
                    outputs=self.model.layers[-2].output
                )
        else:
            # Return specific layer
            layer = self.model.get_layer(layer_name)
            return keras.Model(
                inputs=self.model.input,
                outputs=layer.output
            )
