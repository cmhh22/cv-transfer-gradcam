"""
Utility functions for data loading, preprocessing, and visualization
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional

# Lazy framework imports
try:
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
except ImportError:
    torch = None
    DataLoader = None
    datasets = None
    transforms = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

def prepare_pytorch_dataset(data_dir: str, batch_size: int = 32, 
                            img_size: int = 224, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare PyTorch data loaders
    
    Args:
        data_dir: Path to data directory (should contain train/ and val/ subdirs)
        batch_size: Batch size
        img_size: Image size
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentation for validation
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=train_transform)
    val_dataset = datasets.ImageFolder(f"{data_dir}/val", transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def prepare_tensorflow_dataset(data_dir: str, batch_size: int = 32, 
                               img_size: int = 224) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Prepare TensorFlow datasets
    
    Args:
        data_dir: Path to data directory (should contain train/ and val/ subdirs)
        batch_size: Batch size
        img_size: Image size
        
    Returns:
        Tuple of (train_ds, val_ds)
    """
    # Data augmentation for training
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # No augmentation for validation
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    # Load datasets
    train_ds = train_datagen.flow_from_directory(
        f"{data_dir}/train",
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    val_ds = val_datagen.flow_from_directory(
        f"{data_dir}/val",
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    return train_ds, val_ds

def download_imagenet_labels() -> list:
    """Download ImageNet class labels"""
    import urllib.request
    import json
    
    try:
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        with urllib.request.urlopen(url) as response:
            labels = json.loads(response.read().decode())
        return labels
    except Exception as e:
        print(f"Error downloading labels: {e}")
        return [f"class_{i}" for i in range(1000)]

def plot_training_history(history, save_path: Optional[str] = None):
    """
    Plot training history
    
    Args:
        history: Training history from model.fit()
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Training and Validation Accuracy')
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Training and Validation Loss')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def visualize_predictions(model, images: list, labels: list, class_names: list, 
                         framework: str = 'pytorch', save_path: Optional[str] = None):
    """
    Visualize model predictions on a batch of images
    
    Args:
        model: Trained model
        images: List of images
        labels: True labels
        class_names: List of class names
        framework: 'pytorch' or 'tensorflow'
        save_path: Path to save the visualization
    """
    num_images = len(images)
    cols = 4
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten() if num_images > 1 else [axes]
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        if framework == 'pytorch':
            pred, conf = model.predict(img)
        else:
            pred, conf = model.predict(img)
        
        # Display image
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        # Title with prediction and true label
        true_label = class_names[label] if label < len(class_names) else f"class_{label}"
        title = f"True: {true_label}\nPred: {pred}\nConf: {conf:.2%}"
        color = 'green' if pred == true_label else 'red'
        axes[idx].set_title(title, color=color, fontsize=10)
    
    # Hide extra subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

def load_and_preprocess_image(image_path: str, img_size: int = 224) -> Image.Image:
    """
    Load and preprocess a single image
    
    Args:
        image_path: Path to image
        img_size: Target size
        
    Returns:
        PIL Image
    """
    image = Image.open(image_path).convert('RGB')
    image = image.resize((img_size, img_size))
    return image

def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def count_parameters(model, framework: str = 'pytorch') -> int:
    """
    Count the number of trainable parameters
    
    Args:
        model: Model
        framework: 'pytorch' or 'tensorflow'
        
    Returns:
        Number of trainable parameters
    """
    if framework == 'pytorch':
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return model.count_params()

def create_sample_images(output_dir: str = 'examples', num_images: int = 5):
    """
    Create sample placeholder images for testing
    
    Args:
        output_dir: Directory to save images
        num_images: Number of sample images to create
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    categories = ['cat', 'dog', 'car', 'bird', 'flower']
    
    for i, category in enumerate(categories[:num_images]):
        # Create a random image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Save
        img.save(f"{output_dir}/{category}.jpg")
    
    print(f"Created {num_images} sample images in {output_dir}/")

def setup_directories():
    """Create necessary project directories"""
    directories = ['models', 'data/train', 'data/val', 'examples', 'notebooks', 'docs']
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("Project directories created successfully!")

if __name__ == "__main__":
    # Setup directories and create sample images
    setup_directories()
    create_sample_images()
