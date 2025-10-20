"""
ResNet50V2 and ResNet18 Architectures for Audio Classification
Implementation of ResNet variants optimized for audio spectrograms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18
import warnings

class ResNet50V2(nn.Module):
    """
    ResNet50V2 adapted for audio spectrogram classification
    """
    def __init__(self, num_classes=26, pretrained=True, input_channels=3):
        super(ResNet50V2, self).__init__()
        
        # Load pretrained ResNet50
        if pretrained:
            self.backbone = resnet50(pretrained=True)
        else:
            self.backbone = resnet50(pretrained=False)
        
        # Modify first layer if input channels != 3
        if input_channels != 3:
            original_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            
            # Initialize new conv layer
            if pretrained and input_channels == 1:
                # For grayscale, use mean of RGB weights
                with torch.no_grad():
                    self.backbone.conv1.weight = nn.Parameter(
                        original_conv.weight.mean(dim=1, keepdim=True)
                    )
        
        # Get number of features from ResNet classifier
        num_features = self.backbone.fc.in_features
        
        # Replace classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Store configuration
        self.num_classes = num_classes
        self.input_channels = input_channels
        
    def forward(self, x):
        return self.backbone(x)
    
    def get_feature_maps(self, x):
        """Extract feature maps before final classification"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        return x.flatten(1)


class ResNet18(nn.Module):
    """
    ResNet18 adapted for audio spectrogram classification
    """
    def __init__(self, num_classes=26, pretrained=True, input_channels=3):
        super(ResNet18, self).__init__()
        
        # Load pretrained ResNet18
        if pretrained:
            self.backbone = resnet18(pretrained=True)
        else:
            self.backbone = resnet18(pretrained=False)
        
        # Modify first layer if input channels != 3
        if input_channels != 3:
            original_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            
            # Initialize new conv layer
            if pretrained and input_channels == 1:
                # For grayscale, use mean of RGB weights
                with torch.no_grad():
                    self.backbone.conv1.weight = nn.Parameter(
                        original_conv.weight.mean(dim=1, keepdim=True)
                    )
        
        # Get number of features from ResNet classifier
        num_features = self.backbone.fc.in_features
        
        # Replace classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Store configuration
        self.num_classes = num_classes
        self.input_channels = input_channels
        
    def forward(self, x):
        return self.backbone(x)
    
    def get_feature_maps(self, x):
        """Extract feature maps before final classification"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        return x.flatten(1)


def create_resnet50_v2(num_classes=26, pretrained=True, input_channels=3):
    """
    Factory function to create ResNet50V2 model
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        input_channels: Number of input channels (1 for grayscale, 3 for RGB)
    
    Returns:
        ResNet50V2 model
    """
    return ResNet50V2(
        num_classes=num_classes,
        pretrained=pretrained,
        input_channels=input_channels
    )


def create_resnet18(num_classes=26, pretrained=True, input_channels=3):
    """
    Factory function to create ResNet18 model
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        input_channels: Number of input channels (1 for grayscale, 3 for RGB)
    
    Returns:
        ResNet18 model
    """
    return ResNet18(
        num_classes=num_classes,
        pretrained=pretrained,
        input_channels=input_channels
    )


# For backward compatibility
class FSCResNet50V2(ResNet50V2):
    """Legacy class name for compatibility"""
    pass


class FSCResNet18(ResNet18):
    """Legacy class name for compatibility"""
    pass
