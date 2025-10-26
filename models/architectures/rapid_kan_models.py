"""
Fast-Learning Superior KAN Implementation
Optimized for rapid convergence and high performance
"""

import torch
import torch.nn.functional as F
import math
from .exact_kan_models import KANLinear


class FastLearningKANLinear(KANLinear):
    """
    KANLinear optimized for faster learning while maintaining expressiveness
    """
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=12,            # Moderate grid size for balance
        spline_order=3,          # Keep standard order for stability
        scale_noise=0.01,        # Reduced noise for faster convergence
        scale_base=2.0,          # Strong base activation for faster learning
        scale_spline=1.5,        # Moderate spline contribution
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.01,
        grid_range=[-2, 2],      # Moderate range
        prob_update_grid=0.05,   # Less frequent but more stable updates
        learning_rate_multiplier=1.0
    ):
        self.learning_rate_multiplier = learning_rate_multiplier
        super().__init__(
            in_features, out_features, grid_size, spline_order,
            scale_noise, scale_base, scale_spline, enable_standalone_scale_spline,
            base_activation, grid_eps, grid_range, prob_update_grid
        )
        
        # Optimized initialization for faster learning
        self.fast_initialization()
        
    def fast_initialization(self):
        """Initialization optimized for faster convergence"""
        # Stronger base weight initialization
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5), mode='fan_in')
        self.base_weight.data *= 1.5  # Boost initial weights
        
        # Smart spline weight initialization
        with torch.no_grad():
            # Initialize with identity-like functions for faster learning
            std = 0.2 / math.sqrt(self.in_features)
            self.spline_weight.data.normal_(0, std)
            
            # Add some identity bias to help with early learning
            if self.in_features == self.out_features:
                # For square layers, add identity component
                identity_strength = 0.1
                for i in range(min(self.in_features, self.out_features)):
                    center_idx = self.grid_size // 2
                    if center_idx < self.spline_weight.shape[2]:
                        self.spline_weight.data[i, i, center_idx] += identity_strength
        
        # Optimized spline scaler
        if self.enable_standalone_scale_spline:
            torch.nn.init.constant_(self.spline_scaler, 1.2)


class RapidESC_KAN(torch.nn.Module):
    """
    Rapid-learning KAN optimized for fast convergence to >90% accuracy
    Balances expressiveness with learning speed
    """
    def __init__(self,
                 input_shape,
                 num_classes=26,
                 architecture='efficient',     # 'efficient', 'powerful', 'lightweight'
                 dropout_rate=0.1,
                 batch_norm=True,
                 residual_connections=True):
        super().__init__()
        
        h, w, c = input_shape
        self.architecture = architecture
        self.residual_connections = residual_connections
        
        # Efficient CNN frontend for feature extraction
        self.cnn_frontend = torch.nn.Sequential(
            # Efficient feature extraction
            torch.nn.Conv2d(c, 64, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm2d(64) if batch_norm else torch.nn.Identity(),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(128) if batch_norm else torch.nn.Identity(),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(256) if batch_norm else torch.nn.Identity(),
            torch.nn.ReLU(True),
            torch.nn.AdaptiveAvgPool2d((4, 4))  # Fixed small output
        )
        
        cnn_output_features = 256 * 4 * 4  # 4096 features
        
        # Architecture-specific KAN layers
        if architecture == 'efficient':
            kan_layers = [2048, 512, 128]
            grid_size = 10
        elif architecture == 'powerful':
            kan_layers = [2048, 1024, 256]
            grid_size = 15
        else:  # lightweight
            kan_layers = [1024, 256]
            grid_size = 8
        
        # Input projection to reduce complexity
        self.input_projection = torch.nn.Sequential(
            torch.nn.Linear(cnn_output_features, kan_layers[0]),
            torch.nn.BatchNorm1d(kan_layers[0]) if batch_norm else torch.nn.Identity(),
            torch.nn.ReLU(True),
            torch.nn.Dropout(dropout_rate)
        )
        
        # Build fast-learning KAN layers
        layers_hidden = kan_layers + [num_classes]
        
        self.kan_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        self.residual_projections = torch.nn.ModuleList()
        
        for i, (in_f, out_f) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            # Fast-learning KAN layer
            kan_layer = FastLearningKANLinear(
                in_f, out_f,
                grid_size=grid_size,
                scale_base=2.5,      # Strong base for faster learning
                scale_spline=1.2,    # Moderate spline contribution
                prob_update_grid=0.03,  # Conservative grid updates
                learning_rate_multiplier=2.0 if i < len(layers_hidden) - 2 else 1.0
            )
            
            self.kan_layers.append(kan_layer)
            
            # Batch normalization for faster convergence
            if batch_norm and i < len(layers_hidden) - 2:
                self.batch_norms.append(torch.nn.BatchNorm1d(out_f))
            else:
                self.batch_norms.append(torch.nn.Identity())
            
            # Strategic dropout
            if i < len(layers_hidden) - 2:
                self.dropouts.append(torch.nn.Dropout(dropout_rate))
            else:
                self.dropouts.append(torch.nn.Identity())
            
            # Residual connections for better gradient flow
            if residual_connections and in_f == out_f:
                self.residual_projections.append(torch.nn.Identity())
            elif residual_connections and i < len(layers_hidden) - 2:
                self.residual_projections.append(torch.nn.Linear(in_f, out_f))
            else:
                self.residual_projections.append(None)
        
        # Learning rate scheduling hint
        self.recommended_lr = 0.003  # Higher learning rate for KAN
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"RapidESC_KAN created ({architecture}):")
        print(f"  - KAN Layers: {layers_hidden}")
        print(f"  - Grid Size: {grid_size}")
        print(f"  - Total Parameters: {total_params:,}")
        print(f"  - Recommended LR: {self.recommended_lr}")

    def forward(self, x):
        # Handle input format
        if x.dim() == 4 and x.shape[1] not in [1, 3]:
            x = x.permute(0, 3, 1, 2)
        
        # CNN feature extraction
        x = self.cnn_frontend(x)
        x = x.reshape(x.size(0), -1)  # Use reshape instead of view for safety
        
        # Input projection
        x = self.input_projection(x)
        
        # Fast KAN processing with residual connections
        for i, (kan_layer, batch_norm, dropout, residual_proj) in enumerate(
            zip(self.kan_layers, self.batch_norms, self.dropouts, self.residual_projections)
        ):
            residual = x
            
            # KAN transformation
            x = kan_layer(x)
            
            # Residual connection
            if residual_proj is not None:
                if isinstance(residual_proj, torch.nn.Identity):
                    x = x + residual
                else:
                    x = x + residual_proj(residual)
            
            # Batch norm and dropout
            x = batch_norm(x)
            x = dropout(x)
        
        return x

    def get_optimizer_params(self, base_lr=0.001):
        """Get optimized parameters for different parts of the network"""
        cnn_params = list(self.cnn_frontend.parameters()) + list(self.input_projection.parameters())
        kan_params = []
        other_params = []
        
        for kan_layer in self.kan_layers:
            kan_params.extend([kan_layer.base_weight, kan_layer.spline_weight])
            if hasattr(kan_layer, 'spline_scaler'):
                kan_params.append(kan_layer.spline_scaler)
        
        for batch_norm in self.batch_norms:
            if hasattr(batch_norm, 'weight'):
                other_params.extend([batch_norm.weight, batch_norm.bias])
        
        for residual_proj in self.residual_projections:
            if residual_proj is not None and not isinstance(residual_proj, torch.nn.Identity):
                other_params.extend(list(residual_proj.parameters()))
        
        return [
            {'params': cnn_params, 'lr': base_lr},
            {'params': kan_params, 'lr': base_lr * 3.0},  # Higher LR for KAN layers
            {'params': other_params, 'lr': base_lr * 1.5}
        ]


def create_rapid_kan(input_shape, num_classes=26, performance='efficient'):
    """
    Create Rapid KAN optimized for fast learning to >90% accuracy
    
    Args:
        performance: 'lightweight', 'efficient', 'powerful'
    """
    return RapidESC_KAN(
        input_shape, num_classes,
        architecture=performance,
        dropout_rate=0.15 if performance == 'lightweight' else 0.1,
        batch_norm=True,
        residual_connections=True
    )


def create_training_config_for_rapid_kan():
    """
    Return optimized training configuration for Rapid KAN
    """
    return {
        'learning_rate': 0.003,
        'batch_size': 32,
        'optimizer': 'adamw',
        'weight_decay': 0.01,
        'scheduler': 'cosine',
        'warmup_epochs': 5,
        'patience': 15,
        'factor': 0.5
    }
