"""
Utility functions for data loading, preprocessing, and visualization.
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def prepare_pytorch_dataset(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
) -> Tuple:
    """
    Prepare PyTorch data loaders.

    Args:
        data_dir: Path containing train/ and val/ subdirectories.
        batch_size: Batch size.
        img_size: Image size for resizing.
        num_workers: Number of data-loading workers.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    if transforms is None:
        raise ImportError("PyTorch / torchvision is required: pip install torch torchvision")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(f"{data_dir}/train",
                                         transform=train_transform)
    val_dataset = datasets.ImageFolder(f"{data_dir}/val",
                                       transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)

    return train_loader, val_loader


def prepare_tensorflow_dataset(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
) -> Tuple:
    """
    Prepare TensorFlow datasets using ``image_dataset_from_directory``.

    Args:
        data_dir: Path containing train/ and val/ subdirectories.
        batch_size: Batch size.
        img_size: Image size for resizing.

    Returns:
        Tuple of (train_ds, val_ds).
    """
    if tf is None:
        raise ImportError("TensorFlow is required: pip install tensorflow")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/train",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/val",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    # Normalise pixel values to [0, 1]
    norm = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (norm(x), y))
    val_ds = val_ds.map(lambda x, y: (norm(x), y))

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# ImageNet labels (delegates to the cached version in src modules)
# ---------------------------------------------------------------------------

def download_imagenet_labels() -> list:
    """Download ImageNet class labels (with local file caching)."""
    try:
        from src.pytorch_transfer import get_imagenet_labels
        return get_imagenet_labels()
    except ImportError:
        pass
    try:
        from src.tensorflow_transfer import get_imagenet_labels
        return get_imagenet_labels()
    except ImportError:
        pass

    # Standalone fallback
    import urllib.request
    import json
    try:
        url = ("https://raw.githubusercontent.com/anishathalye/"
               "imagenet-simple-labels/master/imagenet-simple-labels.json")
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return [f"class_{i}" for i in range(1000)]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_training_history(history, save_path: Optional[str] = None):
    """
    Plot training history (accuracy + loss curves).

    Args:
        history: Training history from model.fit() or a dict.
        save_path: Path to save the plot (shows interactively if None).
    """
    h = history.history if hasattr(history, "history") else history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if "accuracy" in h:
        ax1.plot(h["accuracy"], label="Train Accuracy")
    if "val_accuracy" in h:
        ax1.plot(h["val_accuracy"], label="Val Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)
    ax1.set_title("Training and Validation Accuracy")

    if "loss" in h:
        ax2.plot(h["loss"], label="Train Loss")
    if "val_loss" in h:
        ax2.plot(h["val_loss"], label="Val Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)
    ax2.set_title("Training and Validation Loss")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def visualize_predictions(
    model,
    images: list,
    labels: list,
    class_names: list,
    framework: str = "pytorch",
    save_path: Optional[str] = None,
):
    """
    Visualize model predictions on a batch of images.

    Args:
        model: A ``PyTorchTransferModel`` or ``TensorFlowTransferModel`` instance.
        images: List of PIL images.
        labels: True label indices.
        class_names: List of class names.
        framework: 'pytorch' or 'tensorflow' (unused — model.predict() handles it).
        save_path: Path to save the visualisation.
    """
    num_images = len(images)
    cols = min(4, num_images)
    rows = max(1, (num_images + cols - 1) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten() if num_images > 1 else [axes]

    for idx, (img, label) in enumerate(zip(images, labels)):
        results = model.predict(img, top_k=1)
        pred_name, conf = results[0]

        axes[idx].imshow(img)
        axes[idx].axis("off")

        true_label = (class_names[label]
                      if label < len(class_names) else f"class_{label}")
        title = f"True: {true_label}\nPred: {pred_name}\nConf: {conf:.2%}"
        colour = "green" if pred_name == true_label else "red"
        axes[idx].set_title(title, color=colour, fontsize=10)

    for idx in range(num_images, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualisation saved to {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_preprocess_image(image_path: str, img_size: int = 224) -> Image.Image:
    """Load an image and resize to (img_size, img_size)."""
    return Image.open(image_path).convert("RGB").resize((img_size, img_size))


def get_device():
    """Get the best available PyTorch device (CUDA -> MPS -> CPU)."""
    if torch is None:
        raise ImportError("PyTorch is required: pip install torch")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model, framework: str = "pytorch") -> int:
    """Count trainable parameters in a model."""
    if framework == "pytorch":
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model.count_params()


def create_sample_images(output_dir: str = "examples", num_images: int = 5):
    """Create simple placeholder images for testing."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    categories = ["cat", "dog", "car", "bird", "flower"]
    for category in categories[:num_images]:
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(arr).save(f"{output_dir}/{category}.jpg")
    print(f"Created {min(num_images, len(categories))} sample images in {output_dir}/")


def setup_directories():
    """Create necessary project directories."""
    for d in ["models", "data/train", "data/val", "examples", "notebooks", "docs"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("Project directories created successfully!")


if __name__ == "__main__":
    setup_directories()
    create_sample_images()
