import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from config.config import ModelConfig

class WaveletKANBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        
        
        self.high_freq_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        
        self.low_freq_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        
        
        self.detail_extractor = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        
        
        self.freq_enhancement = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.SiLU(),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )
        
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, max(out_channels // 16, 1), 1),
            nn.SiLU(),
            nn.Conv2d(max(out_channels // 16, 1), out_channels, 1),
            nn.Sigmoid()
        )
        
        
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        
        self.wavelet_activation = nn.Sequential(
            nn.SiLU(),
            nn.Dropout2d(dropout)
        )
        
    def forward(self, x):
        identity = self.skip(x)
        
        
        high_freq = self.high_freq_path(x)
        low_freq = self.low_freq_path(x)
        detail = self.detail_extractor(x)
        
        
        combined = high_freq + low_freq + detail
        
        
        freq_weights = self.freq_enhancement(combined)
        enhanced = combined * freq_weights
        
        
        channel_weights = self.channel_attention(enhanced)
        attended = enhanced * channel_weights
        
        
        out = attended + identity
        out = self.wavelet_activation(out)
        
        return out

class HighPerformanceWavKANinspired(nn.Module):
    
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, config: ModelConfig = None):
        super().__init__()
        
        h, w, c = input_shape
        self.config = config or ModelConfig()
        
        self.stem = nn.Sequential(
            nn.Conv2d(c, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2)
        )
        
        dropout = getattr(self.config, 'dropout_rate', 0.1)
        
        self.wavkan_block1 = WaveletKANBlock(32, 64, dropout)
        self.pool1 = nn.MaxPool2d(2)
        
        self.wavkan_block2 = WaveletKANBlock(64, 128, dropout)
        self.pool2 = nn.MaxPool2d(2)
        
        self.wavkan_block3 = WaveletKANBlock(128, 256, dropout)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))
        
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            dummy_out = self.stem(dummy)
            dummy_out = self.pool1(self.wavkan_block1(dummy_out))
            dummy_out = self.pool2(self.wavkan_block2(dummy_out))
            dummy_out = self.pool3(self.wavkan_block3(dummy_out))
            flattened_size = dummy_out.numel()
        
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
        print(f"HighPerformanceWavKAN created with {param_count:,} parameters")
        
    def _init_weights(self, m):
        
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        
        if x.dim() == 4 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
        
        
        x = self.stem(x)
        
        x = self.wavkan_block1(x)
        x = self.pool1(x)
        
        x = self.wavkan_block2(x)
        x = self.pool2(x)
        
        x = self.wavkan_block3(x)
        x = self.pool3(x)
        
        
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

class BasicWavKANinspired(nn.Module):
    """Basic WavKAN model for comparison"""
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, config: ModelConfig = None):
        super().__init__()
        
        h, w, c = input_shape
        self.config = config or ModelConfig()
        
        
        self.features = nn.Sequential(
            
            nn.Conv2d(c, 32, 5, padding=2),
            nn.Conv2d(c, 32, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),
            
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),
            
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            
            dummy_low = self.features[0](dummy)
            dummy_high = self.features[1](dummy)
            dummy_combined = torch.cat([dummy_low, dummy_high], dim=1)
            
            for layer in self.features[2:]:
                dummy_combined = layer(dummy_combined)
            flattened_size = dummy_combined.numel()
        
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
        print(f"BasicWavKAN created with {param_count:,} parameters")
    
    def forward(self, x):
        if x.dim() == 4 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
        
        low_freq = self.features[0](x)
        high_freq = self.features[1](x) 
        combined = torch.cat([low_freq, high_freq], dim=1)
        
        for layer in self.features[2:]:
            combined = layer(combined)
        
        combined = torch.flatten(combined, 1)
        output = self.classifier(combined)
        
        return output

def create_wavkan_inspired_model(config: ModelConfig) -> nn.Module:
    
    input_shape = tuple(config.input_shape)
    num_classes = config.num_classes
    
    model_type = getattr(config, 'model_type', 'high_performance')
    
    if model_type == 'high_performance':
        return HighPerformanceWavKANinspired(input_shape, num_classes, config)
    elif model_type == 'basic':
        return BasicWavKANinspired(input_shape, num_classes, config)
    else:
        raise ValueError(f"Unknown WavKAN model type: {model_type}")

def create_high_performance_wavkan_inspired_model(input_shape: Tuple[int, int, int], num_classes: int) -> nn.Module:
    
    config = ModelConfig(input_shape=input_shape, num_classes=num_classes)
    return create_wavkan_inspired_model(config)
