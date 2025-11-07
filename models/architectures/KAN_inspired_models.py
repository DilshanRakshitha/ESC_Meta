import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any

class ModelConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class ResidualKANBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # KAN-inspired nonlinear activations
        self.kan_activation = nn.Sequential(
            nn.SiLU(),
            nn.Dropout2d(dropout)
        )
        
        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.kan_activation(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.kan_activation(out)
        
        return out

class HighPerformanceKANinspired(nn.Module):
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, config: ModelConfig = None):
        super().__init__()
        
        h, w, c = input_shape
        self.config = config or ModelConfig()
        
        # Multi-scale feature extraction (inspired by EfficientNet + KAN)
        self.stem = nn.Sequential(
            nn.Conv2d(c, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2)
        )
        
        # Residual blocks with KAN activations
        dropout = getattr(self.config, 'dropout_rate', 0.1)
        
        self.block1 = ResidualKANBlock(32, 64, dropout)
        self.pool1 = nn.MaxPool2d(2)
        
        self.block2 = ResidualKANBlock(64, 128, dropout)
        self.pool2 = nn.MaxPool2d(2)
        
        self.block3 = ResidualKANBlock(128, 256, dropout)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))
        
        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            dummy_out = self.stem(dummy)
            dummy_out = self.pool1(self.block1(dummy_out))
            dummy_out = self.pool2(self.block2(dummy_out))
            dummy_out = self.pool3(self.block3(dummy_out))
            flattened_size = dummy_out.numel()
        
        # Advanced classifier with KAN-inspired nonlinearities
        hidden_dim = getattr(self.config, 'hidden_dim', 512)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(flattened_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.apply(self._init_weights)
        
        param_count = sum(p.numel() for p in self.parameters())
        print(f"KAN inspired model created with {param_count:,} parameters")
        
    def _init_weights(self, m):
        """Weight initialization"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Ensure correct input format (B,C,H,W)
        if x.dim() == 4 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
        
        x = self.stem(x)
        
        x = self.block1(x)
        x = self.pool1(x)
        
        x = self.block2(x)
        x = self.pool2(x)
        
        x = self.block3(x)
        x = self.pool3(x)
        
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

class BasicKANinpired(nn.Module):
    """Basic KAN model for comparison"""
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, config: ModelConfig = None):
        super().__init__()
        
        h, w, c = input_shape
        self.config = config or ModelConfig()
        
        # Simple CNN backbone
        self.features = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            dummy_out = self.features(dummy)
            flattened_size = dummy_out.numel()
        
        # KAN-inspired classifier
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        param_count = sum(p.numel() for p in self.parameters())
        print(f"Basic model created with {param_count:,} parameters")
    
    def forward(self, x):
        if x.dim() == 4 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
        
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

def create_kan_inspired_model(config: ModelConfig) -> nn.Module:
    
    input_shape = tuple(config.input_shape)
    num_classes = config.num_classes
    
    model_type = getattr(config, 'model_type', 'high_performance')
    
    if model_type == 'high_performance':
        return HighPerformanceKANinspired(input_shape, num_classes, config)
    elif model_type == 'basic':
        return BasicKANinpired(input_shape, num_classes, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def create_high_performance_kan_inspired(input_shape: Tuple[int, int, int], num_classes: int) -> nn.Module:
    
    config = ModelConfig(input_shape=input_shape, num_classes=num_classes)
    return create_kan_inspired_model(config)
