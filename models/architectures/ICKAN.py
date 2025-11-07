import torch
from torch import nn
import torch.nn.functional as F
from .KAN import KANLinear


class ICKAN(nn.Module):
    ""
    def __init__(self, input_shape=(64, 431, 1), num_classes=26):
        super(ICKAN, self).__init__()
        h, w, c = input_shape
        
        self.conv1 = nn.Conv2d(c, 5, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(5, 5, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.flatten = nn.Flatten()

        fc_input_size = self._get_fc_input_features(h, w, c)
        
        self.kan1 = KANLinear(
            fc_input_size,
            1000,
            grid_size=10,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0,1])
        
        self.kan2 = KANLinear(
            1000,
            num_classes,  # Changed from 20 to num_classes for ESC
            grid_size=10,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0,1])

    def _get_fc_input_features(self, h, w, c):
        """
        Calculate FC input size
        """
        # Use dummy input to calculate feature size
        device = next(self.conv1.parameters()).device if list(self.conv1.parameters()) else 'cpu'
        x = torch.randn(1, c, h, w, device=device)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        return x.size(1)

    def forward(self, x):
        
        # Handle ESC input format
        if x.dim() == 4 and x.shape[1] not in [1, 3]:
            x = x.permute(0, 3, 1, 2)
            
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.kan1(x)
        x = self.kan2(x)
        x = F.log_softmax(x, dim=1)

        return x
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """KAN regularization loss"""
        return (
            self.kan1.regularization_loss(regularize_activation, regularize_entropy) +
            self.kan2.regularization_loss(regularize_activation, regularize_entropy)
        )


# Simple wrapper for different ICKAN variants
class LightICKAN(nn.Module):
    """Lighter version of ICKAN for faster experiments"""
    def __init__(self, input_shape=(64, 431, 1), num_classes=26):
        super().__init__()
        h, w, c = input_shape
        
        # Simplified conv layers
        self.conv1 = nn.Conv2d(c, 3, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        
        # Calculate conv output size
        with torch.no_grad():
            dummy = torch.randn(1, c, h, w)
            dummy = self.maxpool(F.relu(self.conv1(dummy)))
            dummy = self.maxpool(F.relu(self.conv2(dummy)))
            fc_input_size = self.flatten(dummy).size(1)
        
        # Lighter KAN layers
        self.kan1 = KANLinear(
            fc_input_size, 256,
            grid_size=5, spline_order=3,
            scale_noise=0.01, base_activation=nn.SiLU,
            grid_range=[0, 1]
        )
        
        self.kan2 = KANLinear(
            256, num_classes,
            grid_size=5, spline_order=3,
            scale_noise=0.01, base_activation=nn.SiLU,
            grid_range=[0, 1]
        )
    
    def forward(self, x):
        if x.dim() == 4 and x.shape[1] not in [1, 3]:
            x = x.permute(0, 3, 1, 2)
            
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.kan1(x)
        x = self.kan2(x)
        x = F.log_softmax(x, dim=1)
        return x
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return (
            self.kan1.regularization_loss(regularize_activation, regularize_entropy) +
            self.kan2.regularization_loss(regularize_activation, regularize_entropy)
        )


class DeepICKAN(nn.Module):
    """Deeper version of ICKAN"""
    def __init__(self, input_shape=(64, 431, 1), num_classes=26):
        super().__init__()
        h, w, c = input_shape
        
        # More conv layers
        self.conv1 = nn.Conv2d(c, 8, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        
        # Calculate conv output size
        with torch.no_grad():
            dummy = torch.randn(1, c, h, w)
            dummy = self.maxpool(F.relu(self.conv1(dummy)))
            dummy = self.maxpool(F.relu(self.conv2(dummy)))
            dummy = self.maxpool(F.relu(self.conv3(dummy)))
            fc_input_size = self.flatten(dummy).size(1)
        
        # Deeper KAN layers
        self.kan1 = KANLinear(
            fc_input_size, 512,
            grid_size=8, spline_order=3,
            scale_noise=0.01, base_activation=nn.SiLU,
            grid_range=[0, 1]
        )
        
        self.kan2 = KANLinear(
            512, 256,
            grid_size=8, spline_order=3,
            scale_noise=0.01, base_activation=nn.SiLU,
            grid_range=[0, 1]
        )
        
        self.kan3 = KANLinear(
            256, num_classes,
            grid_size=5, spline_order=3,
            scale_noise=0.01, base_activation=nn.SiLU,
            grid_range=[0, 1]
        )
    
    def forward(self, x):
        if x.dim() == 4 and x.shape[1] not in [1, 3]:
            x = x.permute(0, 3, 1, 2)
            
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.kan1(x)
        x = self.kan2(x)
        x = self.kan3(x)
        x = F.log_softmax(x, dim=1)
        return x
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return (
            self.kan1.regularization_loss(regularize_activation, regularize_entropy) +
            self.kan2.regularization_loss(regularize_activation, regularize_entropy) +
            self.kan3.regularization_loss(regularize_activation, regularize_entropy)
        )


# Factory functions for ESC_Meta compatibility
def create_ickan(input_shape, num_classes=26, variant="standard"):
    """Create exact ICKAN models"""
    if variant == "light":
        return LightICKAN(input_shape, num_classes)
    elif variant == "deep":
        return DeepICKAN(input_shape, num_classes)
    else:
        return ICKAN(input_shape, num_classes)

def create_ickan_inspired_models(input_shape, num_classes=26):
    """Backward compatibility function"""
    return ICKAN(input_shape, num_classes)
