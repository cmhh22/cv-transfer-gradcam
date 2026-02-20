# API Documentation

## PyTorchTransferModel

### Initialization

```python
from src.pytorch_transfer import PyTorchTransferModel

model = PyTorchTransferModel(
    model_name='resnet50',    # Architecture name
    num_classes=1000,         # Number of output classes
    pretrained=True           # Load ImageNet weights
)
```

### Methods

#### `predict(image, top_k=5)`

Make prediction on a single image.

**Parameters:**
- `image` (PIL.Image): Input image
- `top_k` (int): Number of top predictions to return (default: 5)

**Returns:**
- List of `(class_name, confidence)` tuples sorted by confidence descending.

**Example:**
```python
from PIL import Image

image = Image.open('cat.jpg')
results = model.predict(image, top_k=5)
for name, conf in results:
    print(f"{name}: {conf:.2%}")
```

#### `get_target_layer_name()`

Return the recommended Grad-CAM target layer name for this architecture.

**Returns:**
- `str` or `None`

#### `freeze_layers(freeze_all=True)`

Freeze model layers for transfer learning.

**Parameters:**
- `freeze_all` (bool): Freeze all layers except final classifier

**Example:**
```python
model.freeze_layers(freeze_all=True)
```

#### `train_model(train_loader, val_loader, epochs=10, lr=0.001, save_path=None)`

Train the model.

**Parameters:**
- `train_loader` (DataLoader): Training data loader
- `val_loader` (DataLoader): Validation data loader
- `epochs` (int): Number of epochs
- `lr` (float): Learning rate
- `save_path` (str): Path to save best model

**Example:**
```python
model.train_model(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20,
    lr=0.001,
    save_path='models/best.pth'
)
```

#### `save(path)` / `load(path)`

Save or load model weights.

**Example:**
```python
model.save('models/my_model.pth')
model.load('models/my_model.pth')
```

---

## TensorFlowTransferModel

### Initialization

```python
from src.tensorflow_transfer import TensorFlowTransferModel

model = TensorFlowTransferModel(
    model_name='ResNet50',
    num_classes=1000,
    input_shape=(224, 224, 3)
)
```

### Methods

#### `predict(image, top_k=5)`

Make prediction on a single image.

**Parameters:**
- `image` (PIL.Image): Input image
- `top_k` (int): Number of top predictions

**Returns:**
- List of `(class_name, confidence)` tuples sorted by confidence descending.

**Example:**
```python
from PIL import Image

image = Image.open('dog.jpg')
results = model.predict(image, top_k=5)
for name, conf in results:
    print(f"{name}: {conf:.2%}")
```

#### `get_target_layer_name()`

Return recommended Grad-CAM target layer for this architecture.

**Returns:**
- `str` or `None`

#### `freeze_layers(freeze_base=True)`

Freeze base model layers.

**Parameters:**
- `freeze_base` (bool): Whether to freeze base layers

#### `compile_model(learning_rate=0.001, loss='categorical_crossentropy')`

Compile the model.

**Parameters:**
- `learning_rate` (float): Learning rate
- `loss` (str): Loss function

**Example:**
```python
model.compile_model(learning_rate=0.001)
```

#### `train_model(train_ds, val_ds, epochs=10, callbacks=None)`

Train the model.

**Parameters:**
- `train_ds`: Training dataset
- `val_ds`: Validation dataset
- `epochs` (int): Number of epochs
- `callbacks` (list): Keras callbacks

**Returns:**
- Training history

**Example:**
```python
history = model.train_model(
    train_ds=train_ds,
    val_ds=val_ds,
    epochs=20
)
```

#### `save(path)` / `load(path)`

Save or load model.

---

## GradCAM

### Initialization

```python
from src.gradcam import GradCAM

gradcam = GradCAM(
    model=model.model,
    framework='pytorch',  # or 'tensorflow'
    target_layer=None     # Auto-detect if None
)
```

### Methods

#### `generate_heatmap(image, target_class=None)`

Generate Grad-CAM heatmap.

**Parameters:**
- `image` (PIL.Image): Input image
- `target_class` (int): Target class index (uses predicted class if None)

**Returns:**
- `numpy.ndarray`: Heatmap (values 0-1)

**Example:**
```python
heatmap = gradcam.generate_heatmap(image, target_class=285)
```

#### `overlay_heatmap(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET)`

Overlay heatmap on original image.

**Parameters:**
- `image` (PIL.Image): Original image
- `heatmap` (numpy.ndarray): Grad-CAM heatmap
- `alpha` (float): Overlay transparency (0-1)
- `colormap` (int): OpenCV colormap

**Returns:**
- `PIL.Image`: Overlayed image

**Example:**
```python
overlay = gradcam.overlay_heatmap(image, heatmap, alpha=0.4)
overlay.save('result.jpg')
```

#### `save_visualization(image, heatmap, output_path, alpha=0.4)`

Save Grad-CAM visualization.

**Example:**
```python
gradcam.save_visualization(image, heatmap, 'gradcam.jpg')
```

---

## Utility Functions

### `prepare_pytorch_dataset(data_dir, batch_size=32, img_size=224, num_workers=4)`

Prepare PyTorch data loaders.

**Returns:**
- Tuple of (train_loader, val_loader)

### `prepare_tensorflow_dataset(data_dir, batch_size=32, img_size=224)`

Prepare TensorFlow datasets.

**Returns:**
- Tuple of (train_ds, val_ds)

### `download_imagenet_labels()`

Download ImageNet class labels.

**Returns:**
- List of 1000 class names

### `plot_training_history(history, save_path=None)`

Plot training history.

### `load_and_preprocess_image(image_path, img_size=224)`

Load and preprocess an image.

**Returns:**
- PIL.Image

---

## Complete Example

```python
from PIL import Image
from src.pytorch_transfer import PyTorchTransferModel
from src.gradcam import GradCAM

# Load model
model = PyTorchTransferModel(model_name='resnet50')

# Load image
image = Image.open('example.jpg')

# Make prediction
results = model.predict(image, top_k=5)
print(f"Top prediction: {results[0][0]} ({results[0][1]:.2%})")

# Generate Grad-CAM
target_layer = model.get_target_layer_name()
gradcam = GradCAM(model.model, framework='pytorch', target_layer=target_layer)
heatmap = gradcam.generate_heatmap(image)
overlay = gradcam.overlay_heatmap(image, heatmap)
overlay.save('gradcam_result.jpg')
gradcam.remove_hooks()
```

---

## Error Handling

All methods include proper error handling. Common errors:

- `ValueError`: Invalid model name or parameters
- `FileNotFoundError`: Model weights file not found
- `RuntimeError`: GPU/memory issues
- `TypeError`: Invalid input type

Example:
```python
try:
    prediction = model.predict(image)
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Runtime error: {e}")
```
