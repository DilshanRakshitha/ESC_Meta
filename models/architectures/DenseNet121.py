import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121
import warnings

class DenseNet121(nn.Module):
    
    def __init__(self, num_classes=26, pretrained=True, input_channels=3):
        super(DenseNet121, self).__init__()
        
        if pretrained:
            self.backbone = densenet121(pretrained=True)
        else:
            self.backbone = densenet121(pretrained=False)
        
        if input_channels != 3:
            original_conv = self.backbone.features.conv0
            self.backbone.features.conv0 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            
            # Initialize new conv layer
            if pretrained and input_channels == 1:
                # For grayscale, use mean of RGB weights
                with torch.no_grad():
                    self.backbone.features.conv0.weight = nn.Parameter(
                        original_conv.weight.mean(dim=1, keepdim=True)
                    )
        
        num_features = self.backbone.classifier.in_features
        
        self.backbone.classifier = nn.Sequential(
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
        features = self.backbone.features(x)
        return F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)


def create_densenet121(num_classes=26, pretrained=True, input_channels=3):
    """
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        input_channels: Number of input channels (1 for grayscale, 3 for RGB)
    
    Returns:
        DenseNet121 model
    """
    return DenseNet121(
        num_classes=num_classes,
        pretrained=pretrained,
        input_channels=input_channels
    )
