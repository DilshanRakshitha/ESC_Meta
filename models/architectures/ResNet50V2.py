"""
ResNet50V2 Model Architecture for FSC
Based on ResNet v2 architecture with pre-activation
Adapted for audio spectrogram classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PreActBottleneck(nn.Module):
    """Pre-activation Bottleneck block for ResNet v2"""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(PreActBottleneck, self).__init__()
        
        # Pre-activation
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                               kernel_size=1, bias=False)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        # Pre-activation
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        
        out = F.relu(self.bn3(out))
        out = self.conv3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        return out


class ResNet50V2(nn.Module):
    """ResNet50 v2 with pre-activation for audio classification"""
    
    def __init__(self, input_channels=3, num_classes=26, dropout_rate=0.2):
        super(ResNet50V2, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Initial convolution (no pre-activation for first layer)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, 
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(2048, num_classes)
        
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """Create a layer with multiple bottleneck blocks"""
        downsample = None
        if stride != 1 or in_channels != out_channels * PreActBottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * PreActBottleneck.expansion,
                         kernel_size=1, stride=stride, bias=False),
            )
        
        layers = []
        layers.append(PreActBottleneck(in_channels, out_channels, stride, downsample))
        
        in_channels = out_channels * PreActBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(PreActBottleneck(in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Final pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class BasicBlock(nn.Module):
    """Basic block for ResNet18"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               padding=1, bias=False)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        return out


class ResNet18(nn.Module):
    """ResNet18 for audio classification"""
    
    def __init__(self, input_channels=3, num_classes=26, dropout_rate=0.2):
        super(ResNet18, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, 
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """Create a layer with multiple basic blocks"""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
            )
        
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Final pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


def create_resnet50_v2(num_classes=26, input_channels=3, dropout_rate=0.2):
    """Create ResNet50 v2 model"""
    logger.info(f"Creating ResNet50 v2 with {num_classes} classes, "
                f"{input_channels} input channels, dropout={dropout_rate}")
    return ResNet50V2(input_channels=input_channels, num_classes=num_classes, 
                      dropout_rate=dropout_rate)


def create_resnet18(num_classes=26, input_channels=3, dropout_rate=0.2):
    """Create ResNet18 model"""
    logger.info(f"Creating ResNet18 with {num_classes} classes, "
                f"{input_channels} input channels, dropout={dropout_rate}")
    return ResNet18(input_channels=input_channels, num_classes=num_classes, 
                    dropout_rate=dropout_rate)
