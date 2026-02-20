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
from typing import Tuple, Optional
import warnings

class PyTorchTransferModel:
    """Transfer Learning with PyTorch pre-trained models"""
    
    def __init__(self, model_name: str = 'resnet50', num_classes: int = 1000, pretrained: bool = True):
        """
        Initialize transfer learning model
        
        Args:
            model_name: Name of the architecture ('resnet50', 'vgg16', etc.)
            num_classes: Number of output classes
            pretrained: Whether to load ImageNet weights
        """
        self.model_name = model_name.lower()
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model(pretrained)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # ImageNet classes (for pretrained models)
        self.class_names = self._load_imagenet_classes()
    
    def _load_model(self, pretrained: bool) -> nn.Module:
        """Load the specified pre-trained model"""
        # Use modern weights API instead of deprecated pretrained=True
        model_loaders = {
            'resnet50': (models.resnet50, models.ResNet50_Weights.DEFAULT),
            'resnet101': (models.resnet101, models.ResNet101_Weights.DEFAULT),
            'vgg16': (models.vgg16, models.VGG16_Weights.DEFAULT),
            'vgg19': (models.vgg19, models.VGG19_Weights.DEFAULT),
            'efficientnetb0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
            'mobilenetv2': (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
        }

        if self.model_name not in model_loaders:
            raise ValueError(f"Model {self.model_name} not supported. Choose from: {list(model_loaders.keys())}")

        loader_fn, default_weights = model_loaders[self.model_name]
        weights = default_weights if pretrained else None
        model = loader_fn(weights=weights)

        # Replace final classification layer if num_classes differs from ImageNet
        if self.num_classes != 1000:
            if self.model_name.startswith('resnet'):
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            elif self.model_name.startswith('vgg'):
                model.classifier[6] = nn.Linear(4096, self.num_classes)
            else:  # efficientnet, mobilenet
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)

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
        self.model = self._load_model(pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()
    
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
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get top prediction
        top_prob, top_class = torch.topk(probabilities, top_k)
        
        if top_k == 1:
            pred_class = top_class.item()
            pred_prob = top_prob.item()
            pred_name = self.class_names[pred_class] if pred_class < len(self.class_names) else f"class_{pred_class}"
            return pred_name, pred_prob
        else:
            results = []
            for i in range(top_k):
                pred_class = top_class[i].item()
                pred_prob = top_prob[i].item()
                pred_name = self.class_names[pred_class] if pred_class < len(self.class_names) else f"class_{pred_class}"
                results.append((pred_name, pred_prob))
            return results
    
    def freeze_layers(self, freeze_all: bool = True):
        """Freeze model layers for fine-tuning"""
        if freeze_all:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Unfreeze final layer
        if hasattr(self.model, 'fc'):
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'classifier'):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
    
    def train_model(self, train_loader, val_loader, epochs: int = 10, 
                   lr: float = 0.001, save_path: Optional[str] = None):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            lr: Learning rate
            save_path: Path to save the best model
        """
        self.model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        best_acc = 0.0
        
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            print('-' * 10)
            
            # Training phase
            self.model.train()
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)
            
            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_corrects = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    val_corrects += torch.sum(preds == labels.data)
            
            val_loss = val_loss / len(val_loader.dataset)
            val_acc = val_corrects.double() / len(val_loader.dataset)
            
            print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
            
            # Save best model
            if val_acc > best_acc and save_path:
                best_acc = val_acc
                self.save(save_path)
                print(f'Best model saved with accuracy: {best_acc:.4f}')
            
            scheduler.step()
            print()
        
        print(f'Training complete. Best validation accuracy: {best_acc:.4f}')
    
    def save(self, path: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {path}")
    
    def get_feature_extractor(self, layer_name: Optional[str] = None):
        """Get intermediate features from a specific layer"""
        if layer_name is None:
            # Return features before final layer
            if hasattr(self.model, 'fc'):
                return nn.Sequential(*list(self.model.children())[:-1])
            elif hasattr(self.model, 'classifier'):
                return nn.Sequential(*list(self.model.children())[:-1])
        else:
            # Return specific layer
            return dict(self.model.named_modules())[layer_name]
