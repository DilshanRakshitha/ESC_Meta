"""
InceptionV3 Architecture for Audio Classification
Inception Networks adapted for spectrograms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3
import warnings

class InceptionV3(nn.Module):
    """
    InceptionV3 adapted for audio spectrogram classification
    """
    def __init__(self, num_classes=26, pretrained=True, input_channels=3, dropout_rate=0.5):
        super(InceptionV3, self).__init__()
        
        # Load pretrained InceptionV3
        # Note: newer torchvision versions force aux_logits=True for pretrained models
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if pretrained:
                self.backbone = inception_v3(weights='IMAGENET1K_V1')  # aux_logits=True by default
            else:
                self.backbone = inception_v3(weights=None, aux_logits=False)
        
        # Modify first layer if input channels != 3
        if input_channels != 3:
            original_conv = self.backbone.Conv2d_1a_3x3.conv
            self.backbone.Conv2d_1a_3x3.conv = nn.Conv2d(
                input_channels, 32, kernel_size=3, stride=2, bias=False
            )
            
            # Initialize new conv layer
            if pretrained and input_channels == 1:
                # For grayscale, use mean of RGB weights
                with torch.no_grad():
                    self.backbone.Conv2d_1a_3x3.conv.weight = nn.Parameter(
                        original_conv.weight.mean(dim=1, keepdim=True)
                    )
        
        # Get number of features from InceptionV3 classifier
        num_features = self.backbone.fc.in_features
        
        # Replace classifier with custom head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.25),
            nn.Linear(512, num_classes)
        )
        
        # Store configuration
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.dropout_rate = dropout_rate
        
        # Disable auxiliary classifier during training
        self.training_mode = True
        
    def forward(self, x):
        # InceptionV3 requires input size to be at least 299x299
        # Resize if needed
        if x.shape[-1] < 299 or x.shape[-2] < 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Handle auxiliary logits output
        if self.training and hasattr(self.backbone, 'AuxLogits'):
            # During training, backbone returns (main_output, aux_output)
            output = self.backbone(x)
            if isinstance(output, tuple):
                return output[0]  # Return only main output
            return output
        else:
            # During evaluation, backbone returns only main output
            self.backbone.eval()
            return self.backbone(x)
    
    def get_feature_maps(self, x):
        """Extract feature maps before final classification"""
        # Resize if needed
        if x.shape[-1] < 299 or x.shape[-2] < 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Forward through features
        x = self.backbone.Conv2d_1a_3x3(x)
        x = self.backbone.Conv2d_2a_3x3(x)
        x = self.backbone.Conv2d_2b_3x3(x)
        x = self.backbone.maxpool1(x)
        x = self.backbone.Conv2d_3b_1x1(x)
        x = self.backbone.Conv2d_4a_3x3(x)
        x = self.backbone.maxpool2(x)
        x = self.backbone.Mixed_5b(x)
        x = self.backbone.Mixed_5c(x)
        x = self.backbone.Mixed_5d(x)
        x = self.backbone.Mixed_6a(x)
        x = self.backbone.Mixed_6b(x)
        x = self.backbone.Mixed_6c(x)
        x = self.backbone.Mixed_6d(x)
        x = self.backbone.Mixed_6e(x)
        x = self.backbone.Mixed_7a(x)
        x = self.backbone.Mixed_7b(x)
        x = self.backbone.Mixed_7c(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = F.dropout(x, training=self.training)
        x = torch.flatten(x, 1)
        
        return x


def create_inception_v3(num_classes=26, pretrained=True, input_channels=3, dropout_rate=0.5):
    """
    Factory function to create InceptionV3 model
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        input_channels: Number of input channels (1 for grayscale, 3 for RGB)
        dropout_rate: Dropout rate for regularization
    
    Returns:
        InceptionV3 model
    """
    return InceptionV3(
        num_classes=num_classes,
        pretrained=pretrained,
        input_channels=input_channels,
        dropout_rate=dropout_rate
    )


# For backward compatibility
class FSCInceptionV3(InceptionV3):
    """Legacy class name for compatibility"""
    pass
