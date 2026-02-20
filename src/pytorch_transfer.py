"""
PyTorch Transfer Learning Module
Implements transfer learning with pre-trained models
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional, List, Dict
import json
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# ImageNet labels cache (download once → reuse across all instances)
# ---------------------------------------------------------------------------
_IMAGENET_LABELS: Optional[list] = None


def get_imagenet_labels() -> list:
    """Load ImageNet class names with local file caching."""
    global _IMAGENET_LABELS
    if _IMAGENET_LABELS is not None:
        return _IMAGENET_LABELS

    cache_path = Path(__file__).parent.parent / "data" / "imagenet_labels.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Try local file first
    if cache_path.exists():
        try:
            _IMAGENET_LABELS = json.loads(cache_path.read_text())
            return _IMAGENET_LABELS
        except Exception:
            pass

    # Download & persist
    try:
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        with urllib.request.urlopen(url, timeout=10) as resp:
            _IMAGENET_LABELS = json.loads(resp.read().decode())
        cache_path.write_text(json.dumps(_IMAGENET_LABELS))
    except Exception:
        _IMAGENET_LABELS = [f"class_{i}" for i in range(1000)]

    return _IMAGENET_LABELS


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PyTorchTransferModel:
    """Transfer Learning with PyTorch pre-trained models."""

    # Class-level weight cache so weights download only once per session
    _weight_cache: Dict[str, "nn.Module"] = {}

    MODEL_REGISTRY = {
        'resnet50':       (models.resnet50,       models.ResNet50_Weights.DEFAULT),
        'resnet101':      (models.resnet101,      models.ResNet101_Weights.DEFAULT),
        'vgg16':          (models.vgg16,          models.VGG16_Weights.DEFAULT),
        'vgg19':          (models.vgg19,          models.VGG19_Weights.DEFAULT),
        'efficientnetb0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
        'mobilenetv2':    (models.mobilenet_v2,   models.MobileNet_V2_Weights.DEFAULT),
    }

    # Best Grad-CAM target layers per architecture
    TARGET_LAYERS = {
        'resnet50':       'layer4',
        'resnet101':      'layer4',
        'vgg16':          'features.28',
        'vgg19':          'features.34',
        'efficientnetb0': 'features.8',
        'mobilenetv2':    'features.18',
    }

    def __init__(self, model_name: str = 'resnet50', num_classes: int = 1000,
                 pretrained: bool = True):
        self.model_name = model_name.lower()
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.model_name not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Model '{self.model_name}' not supported. "
                f"Choose from: {list(self.MODEL_REGISTRY.keys())}"
            )

        self.model = self._load_model(pretrained)
        self.model.to(self.device).eval()

        # Standard ImageNet preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.class_names = get_imagenet_labels()

    # ------------------------------------------------------------------ #
    def _load_model(self, pretrained: bool) -> nn.Module:
        """Load model; uses an in-memory cache to avoid re-downloading."""
        cache_key = f"{self.model_name}_{pretrained}_{self.num_classes}"
        if cache_key in self._weight_cache:
            import copy
            return copy.deepcopy(self._weight_cache[cache_key])

        loader_fn, default_w = self.MODEL_REGISTRY[self.model_name]
        weights = default_w if pretrained else None
        model = loader_fn(weights=weights)

        if self.num_classes != 1000:
            if self.model_name.startswith('resnet'):
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            elif self.model_name.startswith('vgg'):
                model.classifier[6] = nn.Linear(4096, self.num_classes)
            else:
                model.classifier[1] = nn.Linear(
                    model.classifier[1].in_features, self.num_classes)

        self._weight_cache[cache_key] = model
        return model

    def load_pretrained(self):
        """Reload pretrained ImageNet weights."""
        self.model = self._load_model(pretrained=True)
        self.model.to(self.device).eval()

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
        image = image.convert('RGB')
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.nn.functional.softmax(logits[0], dim=0)

        top_p, top_i = torch.topk(probs, min(top_k, len(probs)))
        results = []
        for p, i in zip(top_p, top_i):
            idx = i.item()
            name = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
            results.append((name, round(p.item(), 5)))
        return results

    def get_target_layer_name(self) -> Optional[str]:
        """Return the recommended Grad-CAM target layer for this arch."""
        return self.TARGET_LAYERS.get(self.model_name)

    # ------------------------------------------------------------------ #
    # Training helpers
    # ------------------------------------------------------------------ #
    def freeze_layers(self, freeze_all: bool = True):
        if freeze_all:
            for p in self.model.parameters():
                p.requires_grad = False
        if hasattr(self.model, 'fc'):
            for p in self.model.fc.parameters():
                p.requires_grad = True
        elif hasattr(self.model, 'classifier'):
            for p in self.model.classifier.parameters():
                p.requires_grad = True

    def train_model(self, train_loader, val_loader, epochs: int = 10,
                    lr: float = 0.001, save_path: Optional[str] = None):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        best_acc = 0.0

        for epoch in range(epochs):
            # Train
            self.model.train()
            loss_sum, correct, total = 0.0, 0, 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = criterion(out, y)
                loss.backward(); optimizer.step()
                loss_sum += loss.item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += x.size(0)
            print(f"Epoch {epoch+1}/{epochs}  "
                  f"Train loss={loss_sum/total:.4f}  acc={correct/total:.4f}", end="  ")

            # Val
            self.model.eval()
            v_loss, v_correct, v_total = 0.0, 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.model(x)
                    loss = criterion(out, y)
                    v_loss += loss.item() * x.size(0)
                    v_correct += (out.argmax(1) == y).sum().item()
                    v_total += x.size(0)
            v_acc = v_correct / v_total
            print(f"Val loss={v_loss/v_total:.4f}  acc={v_acc:.4f}")

            if v_acc > best_acc and save_path:
                best_acc = v_acc
                self.save(save_path)
            scheduler.step()

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
