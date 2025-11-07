import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from config.config import ModelConfig

class EnhancedICKANBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        
        self.involution = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, max(out_channels // 16, 1), 1),
            nn.SiLU(),
            nn.Conv2d(max(out_channels // 16, 1), out_channels, 1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        self.kan_activation = nn.Sequential(
            nn.SiLU(),
            nn.Dropout2d(dropout)
        )
        
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.involution(x)
        
        ca = self.channel_attention(out)
        out = out * ca
        
        avg_pool = torch.mean(out, dim=1, keepdim=True)
        max_pool, _ = torch.max(out, dim=1, keepdim=True)
        sa_input = torch.cat([avg_pool, max_pool], dim=1)
        sa = self.spatial_attention(sa_input)
        out = out * sa
        
        out += identity
        out = self.kan_activation(out)
        
        return out

class HighPerformanceICKANinspired(nn.Module):
    
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
        
        self.ickan_block1 = EnhancedICKANBlock(32, 64, dropout)
        self.pool1 = nn.MaxPool2d(2)
        
        self.ickan_block2 = EnhancedICKANBlock(64, 128, dropout)
        self.pool2 = nn.MaxPool2d(2)
        
        self.ickan_block3 = EnhancedICKANBlock(128, 256, dropout)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))
        
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            dummy_out = self.stem(dummy)
            dummy_out = self.pool1(self.ickan_block1(dummy_out))
            dummy_out = self.pool2(self.ickan_block2(dummy_out))
            dummy_out = self.pool3(self.ickan_block3(dummy_out))
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
        print(f"HighPerformanceICKAN created with {param_count:,} parameters")
        
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
        
        x = self.ickan_block1(x)
        x = self.pool1(x)
        
        x = self.ickan_block2(x)
        x = self.pool2(x)
        
        x = self.ickan_block3(x)
        x = self.pool3(x)
        
        
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

class BasicICKANinspired(nn.Module):
    
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, config: ModelConfig = None):
        super().__init__()
        
        h, w, c = input_shape
        self.config = config or ModelConfig()
        
        
        self.features = nn.Sequential(
            
            nn.Conv2d(c, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),
            
            
            nn.Conv2d(32, 64, 1), 
            nn.Conv2d(64, 64, 3, padding=1, groups=64),
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
            dummy_out = self.features(dummy)
            flattened_size = dummy_out.numel()
        
        
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
        print(f"BasicICKAN created with {param_count:,} parameters")
    
    def forward(self, x):
        if x.dim() == 4 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
        
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

class InvolutionLayer(nn.Module):
    
    def __init__(self, channels: int, kernel_size: int = 7, stride: int = 1, 
                 reduction_ratio: int = 4):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduced_channels = channels // reduction_ratio
        
        self.kernel_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size)),
            nn.Conv2d(channels, reduced_channels, 1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, kernel_size * kernel_size, 1)
        )
        
        self.unfold = nn.Unfold(kernel_size, padding=kernel_size // 2, stride=stride)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        
        kernels = self.kernel_gen(x)
        kernels = kernels.view(B, self.kernel_size * self.kernel_size, H // self.stride, W // self.stride)
        
        
        unfolded = self.unfold(x)
        unfolded = unfolded.view(B, C, self.kernel_size * self.kernel_size, H // self.stride, W // self.stride)
        
        output = (unfolded * kernels.unsqueeze(1)).sum(dim=2)
        
        return output

def create_ickan_inspired_models(config: ModelConfig) -> nn.Module:
    
    input_shape = tuple(config.input_shape)
    num_classes = config.num_classes
    
    model_type = getattr(config, 'model_type', 'high_performance')
    
    if model_type == 'high_performance':
        return HighPerformanceICKANinspired(input_shape, num_classes, config)
    elif model_type == 'basic':
        return BasicICKANinspired(input_shape, num_classes, config)
    else:
        raise ValueError(f"Unknown ICKAN model type: {model_type}")

def create_high_performance_ickan_inspired_model(input_shape: Tuple[int, int, int], num_classes: int) -> nn.Module:
    
    config = ModelConfig(input_shape=input_shape, num_classes=num_classes)
    return create_ickan_inspired_models(config)
