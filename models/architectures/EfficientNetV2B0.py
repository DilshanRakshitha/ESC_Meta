import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s
import warnings

class EfficientNetV2B0(nn.Module):

    def __init__(self, num_classes=26, pretrained=True, input_channels=3, dropout_rate=0.2):
        super(EfficientNetV2B0, self).__init__()
        
        if pretrained:
            try:
                self.backbone = efficientnet_v2_s(pretrained=True)
            except:
                warnings.warn("Pretrained EfficientNetV2 not available, using random initialization")
                self.backbone = efficientnet_v2_s(pretrained=False)
        else:
            self.backbone = efficientnet_v2_s(pretrained=False)
        
        # Modify first layer if input channels != 3
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
            
            # Initialize new conv layer
            if pretrained and input_channels == 1:
                # For grayscale, use mean of RGB weights
                with torch.no_grad():
                    self.backbone.features[0][0].weight = nn.Parameter(
                        original_conv.weight.mean(dim=1, keepdim=True)
                    )
        
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.25),
            nn.Linear(256, num_classes)
        )
        
        # Store configuration
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.dropout_rate = dropout_rate
        
    def forward(self, x):
        return self.backbone(x)
    
    def get_feature_maps(self, x):
        
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)


def create_efficientnet_v2_b0(num_classes=26, pretrained=True, input_channels=3, dropout_rate=0.2):
    """
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        input_channels: Number of input channels (1 for grayscale, 3 for RGB)
        dropout_rate: Dropout rate for regularization
    
    Returns:
        EfficientNetV2B0 model
    """
    return EfficientNetV2B0(
        num_classes=num_classes,
        pretrained=pretrained,
        input_channels=input_channels,
        dropout_rate=dropout_rate
    )
