import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
import warnings

class MobileNetV3Small(nn.Module):
    
    def __init__(self, num_classes=26, pretrained=True, input_channels=3, dropout_rate=0.2):
        super(MobileNetV3Small, self).__init__()
        
        if pretrained:
            self.backbone = mobilenet_v3_small(pretrained=True)
        else:
            self.backbone = mobilenet_v3_small(pretrained=False)
        
        if input_channels != 3:
            original_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            
            
            if pretrained and input_channels == 1:
                
                with torch.no_grad():
                    self.backbone.features[0][0].weight = nn.Parameter(
                        original_conv.weight.mean(dim=1, keepdim=True)
                    )
        
        num_features = self.backbone.classifier[0].in_features
        
        
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.dropout_rate = dropout_rate
        
    def forward(self, x):
        return self.backbone(x)
    
    def get_feature_maps(self, x):
        
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def count_parameters(self):
        
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_mobilenet_v3_small(num_classes=26, pretrained=True, input_channels=3, dropout_rate=0.2):
    """
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        input_channels: Number of input channels (1 for grayscale, 3 for RGB)
        dropout_rate: Dropout rate for regularization
    
    Returns:
        MobileNetV3Small model
    """
    return MobileNetV3Small(
        num_classes=num_classes,
        pretrained=pretrained,
        input_channels=input_channels,
        dropout_rate=dropout_rate
    )

class MobileNetV3Large(nn.Module):
    
    def __init__(self, num_classes=26, pretrained=True, input_channels=3, dropout_rate=0.2):
        super(MobileNetV3Large, self).__init__()
        from torchvision.models import mobilenet_v3_large
        
        self.backbone = mobilenet_v3_large(pretrained=pretrained)
        
        if input_channels != 3:
            original_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            if pretrained and input_channels == 1:
                with torch.no_grad():
                    self.backbone.features[0][0].weight = nn.Parameter(
                        original_conv.weight.mean(dim=1, keepdim=True)
                    )
        
        num_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)


def create_mobilenet_v3_large(num_classes=26, pretrained=True, input_channels=3, dropout_rate=0.2):
    
    return MobileNetV3Large(num_classes, pretrained, input_channels, dropout_rate)
