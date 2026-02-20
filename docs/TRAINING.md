# Training Guide

## Preparing Your Dataset

### Directory Structure

Organize your dataset in the following structure:

```
data/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── image1.jpg
│   │   └── ...
│   └── ...
└── val/
    ├── class1/
    │   ├── image1.jpg
    │   └── ...
    ├── class2/
    └── ...
```

### Data Recommendations

- **Image Format**: JPEG or PNG
- **Image Size**: At least 224x224 pixels
- **Training Samples**: At least 100 images per class
- **Validation Split**: 20% of total data
- **Balance**: Try to maintain similar number of samples per class

## Training with PyTorch

### Basic Training

```python
from src.pytorch_transfer import PyTorchTransferModel
from src.utils import prepare_pytorch_dataset

# Prepare data
train_loader, val_loader = prepare_pytorch_dataset(
    data_dir='data',
    batch_size=32,
    img_size=224
)

# Initialize model
model = PyTorchTransferModel(
    model_name='resnet50',
    num_classes=10,  # Your number of classes
    pretrained=True
)

# Freeze base layers (transfer learning)
model.freeze_layers(freeze_all=True)

# Train
model.train_model(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20,
    lr=0.001,
    save_path='models/best_model.pth'
)
```

### Fine-tuning

For better results, unfreeze some layers after initial training:

```python
# Unfreeze all layers
for param in model.model.parameters():
    param.requires_grad = True

# Train with lower learning rate
model.train_model(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    lr=0.0001,
    save_path='models/finetuned_model.pth'
)
```

## Training with TensorFlow

### Basic Training

```python
from src.tensorflow_transfer import TensorFlowTransferModel
from src.utils import prepare_tensorflow_dataset

# Prepare data
train_ds, val_ds = prepare_tensorflow_dataset(
    data_dir='data',
    batch_size=32,
    img_size=224
)

# Initialize model
model = TensorFlowTransferModel(
    model_name='ResNet50',
    num_classes=10,
    input_shape=(224, 224, 3)
)

# Freeze base layers
model.freeze_layers(freeze_base=True)

# Compile
model.compile_model(learning_rate=0.001)

# Train
history = model.train_model(
    train_ds=train_ds,
    val_ds=val_ds,
    epochs=20
)

# Save
model.save('models/best_model.h5')
```

### Fine-tuning

```python
# Unfreeze base layers
model.freeze_layers(freeze_base=False)

# Recompile with lower learning rate
model.compile_model(learning_rate=0.0001)

# Continue training
history = model.train_model(
    train_ds=train_ds,
    val_ds=val_ds,
    epochs=10
)
```

## Hyperparameter Tuning

### Learning Rate

- **Initial training**: 0.001 - 0.01
- **Fine-tuning**: 0.0001 - 0.001
- Use learning rate scheduling for better convergence

### Batch Size

- **Small GPU (4-8GB)**: 16-32
- **Large GPU (16GB+)**: 64-128
- Larger batch sizes may improve generalization

### Epochs

- **Transfer learning**: 10-20 epochs
- **Fine-tuning**: 5-10 epochs
- Use early stopping to prevent overfitting

### Data Augmentation

Recommended augmentations:
- Random horizontal flip
- Random rotation (±15°)
- Random crop/zoom (0.8-1.2x)
- Color jitter (brightness, contrast, saturation)

## Model Selection Guide

| Model | Speed | Accuracy | GPU Memory | Use Case |
|-------|-------|----------|------------|----------|
| MobileNetV2 | Fast | Good | Low | Mobile, Edge devices |
| EfficientNetB0 | Fast | Good | Low | Balanced performance |
| ResNet50 | Medium | Very Good | Medium | General purpose |
| ResNet101 | Slow | Excellent | High | High accuracy needed |
| VGG16/19 | Slow | Very Good | High | Feature extraction |

## Monitoring Training

### Key Metrics

1. **Training Loss**: Should steadily decrease
2. **Validation Loss**: Should decrease but watch for overfitting
3. **Training Accuracy**: Should increase
4. **Validation Accuracy**: Primary metric for model selection

### Overfitting Signs

- Validation loss increases while training loss decreases
- Large gap between training and validation accuracy
- Model performs poorly on new data

### Solutions

- Increase data augmentation
- Add dropout layers
- Reduce model complexity
- Get more training data
- Use regularization (L1/L2)

## Best Practices

1. **Start Simple**: Begin with transfer learning before fine-tuning
2. **Monitor Closely**: Use TensorBoard or similar tools
3. **Save Checkpoints**: Save models at regular intervals
4. **Validate Often**: Check validation metrics frequently
5. **Test Thoroughly**: Test on held-out test set
6. **Document Everything**: Keep track of hyperparameters

## Troubleshooting

### Out of Memory

- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training
- Use smaller model

### Slow Training

- Increase batch size
- Use data loading optimization (num_workers)
- Enable GPU acceleration
- Use mixed precision training

### Poor Performance

- Check data preprocessing
- Verify data augmentation
- Ensure proper normalization
- Try different architectures
- Increase training data

## Example Training Scripts

See `notebooks/` directory for complete training examples:
- `01_pytorch_tutorial.ipynb`
- `02_tensorflow_tutorial.ipynb`

## Command Line Training

```bash
# PyTorch
python src/train_pytorch.py \
    --model resnet50 \
    --data data/ \
    --epochs 20 \
    --batch-size 32 \
    --lr 0.001 \
    --output models/model.pth

# TensorFlow
python src/train_tensorflow.py \
    --model ResNet50 \
    --data data/ \
    --epochs 20 \
    --batch-size 32 \
    --lr 0.001 \
    --output models/model.h5
```
