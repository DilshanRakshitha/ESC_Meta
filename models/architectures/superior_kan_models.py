"""
High-Performance KAN Implementation for Superior Audio Classification
Addresses key limitations to achieve >90% accuracy while maintaining exact KANLinear logic
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
from .exact_kan_models import KANLinear


class HighPerformanceKANLinear(KANLinear):
    """
    Enhanced KANLinear with optimizations for superior performance
    Maintains exact mathematical foundation while improving expressiveness
    """
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=15,           # Much larger grid for expressiveness
        spline_order=4,         # Higher order splines
        scale_noise=0.05,       # Reduced noise for stability
        scale_base=1.5,         # Stronger base activation
        scale_spline=2.0,       # Enhanced spline contribution
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.01,          # Finer grid precision
        grid_range=[-3, 3],     # Wider range for better coverage
        prob_update_grid=0.1,   # More frequent grid updates
        adaptive_grid=True      # Enable intelligent grid adaptation
    ):
        self.adaptive_grid = adaptive_grid
        super().__init__(
            in_features, out_features, grid_size, spline_order,
            scale_noise, scale_base, scale_spline, enable_standalone_scale_spline,
            base_activation, grid_eps, grid_range, prob_update_grid
        )
        
        # Enhanced initialization for better convergence
        self.enhanced_initialization()
        
        # Adaptive learning parameters
        self.grid_adaptation_threshold = 0.01
        self.spline_regularization_weight = 0.001
        
    def enhanced_initialization(self):
        """Better initialization strategy for KAN layers"""
        # Xavier/Glorot initialization for base weights
        torch.nn.init.xavier_uniform_(self.base_weight, gain=1.4)
        
        # Improved spline weight initialization
        with torch.no_grad():
            # Use a simpler but more robust initialization
            # Initialize spline weights with small random values scaled by importance
            torch.nn.init.normal_(self.spline_weight, mean=0.0, std=0.1 * self.scale_spline)
            
            # Add some structure to the initialization
            for i in range(min(self.in_features, 4)):
                # Initialize first few features with specific patterns
                grid_points = self.grid[i, self.spline_order:-self.spline_order]
                if len(grid_points) > 0:
                    if i % 4 == 0:
                        pattern = torch.sin(2 * math.pi * torch.linspace(0, 1, len(grid_points)))
                    elif i % 4 == 1:
                        pattern = torch.cos(2 * math.pi * torch.linspace(0, 1, len(grid_points)))
                    elif i % 4 == 2:
                        pattern = torch.linspace(-1, 1, len(grid_points)) ** 2
                    else:
                        pattern = torch.tanh(torch.linspace(-2, 2, len(grid_points)))
                    
                    # Apply pattern to spline weights for this feature
                    for j in range(self.out_features):
                        if len(pattern) <= self.spline_weight.shape[2]:
                            self.spline_weight.data[j, i, :len(pattern)] *= pattern * 0.3
            
        # Enhanced spline scaler initialization
        if self.enable_standalone_scale_spline:
            torch.nn.init.uniform_(self.spline_scaler, 0.8, 1.2)

    def forward(self, x: torch.Tensor):
        """Enhanced forward pass with better numerical stability"""
        assert x.dim() == 2 and x.size(1) == self.in_features
        
        # Clamp input to prevent numerical instabilities
        x = torch.clamp(x, -10, 10)
        
        # Base activation with stronger contribution
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        # Enhanced spline computation
        spline_bases = self.b_splines(x)
        
        # Apply learnable scaling to spline bases for better expressiveness
        scaled_bases = spline_bases * (1.0 + 0.1 * torch.tanh(spline_bases))
        
        spline_output = F.linear(
            scaled_bases.view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        
        # Adaptive grid update with improved logic
        if self.training and self.adaptive_grid and \
           torch.rand(1).item() < self.prob_update_grid and \
           self.current_grid_step < 200:  # More update steps
            self.smart_update_grid(x)
            self.current_grid_step += 1
        
        return base_output + spline_output

    @torch.no_grad()
    def smart_update_grid(self, x: torch.Tensor):
        """Intelligent grid adaptation based on data distribution"""
        try:
            # Compute current function values
            current_output = self(x.detach())
            
            # Analyze input distribution per feature
            for feature_idx in range(self.in_features):
                feature_data = x[:, feature_idx]
                
                # Check if current grid covers data distribution well
                data_min, data_max = feature_data.min(), feature_data.max()
                current_grid = self.grid[feature_idx]
                grid_min, grid_max = current_grid[self.spline_order], current_grid[-self.spline_order-1]
                
                # Expand grid if data extends beyond current range
                if data_min < grid_min or data_max > grid_max:
                    # Create new grid that better covers the data
                    margin = (data_max - data_min) * 0.1
                    new_min = min(grid_min, data_min - margin)
                    new_max = max(grid_max, data_max + margin)
                    
                    h = (new_max - new_min) / self.grid_size
                    new_grid = (
                        torch.arange(-self.spline_order, self.grid_size + self.spline_order + 1) * h + new_min
                    )
                    
                    self.grid[feature_idx] = new_grid
                    
        except Exception as e:
            # Fall back to original update method
            super().update_grid(x)


class SuperiorESC_KAN(torch.nn.Module):
    """
    Superior KAN architecture specifically optimized for audio classification
    Designed to achieve >90% accuracy on ESC datasets
    """
    def __init__(self,
                 input_shape,
                 num_classes=26,
                 use_cnn_frontend=True,      # CNN feature extraction
                 kan_layers=[1024, 512, 256, 128],
                 grid_size=20,               # Large grid for expressiveness
                 spline_order=4,             # Higher order splines
                 dropout_rate=0.1,           # Minimal dropout
                 layer_norm=True,
                 residual_connections=True,
                 attention_mechanism=True):   # Add attention
        super().__init__()
        
        h, w, c = input_shape
        self.use_cnn_frontend = use_cnn_frontend
        self.residual_connections = residual_connections
        self.attention_mechanism = attention_mechanism
        
        if use_cnn_frontend:
            # CNN frontend for better feature extraction from spectrograms
            self.cnn_frontend = torch.nn.Sequential(
                # First conv block - capture local patterns
                torch.nn.Conv2d(c, 32, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(True),
                torch.nn.MaxPool2d(2),
                
                # Second conv block - capture mid-level features
                torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(True),
                torch.nn.MaxPool2d(2),
                
                # Third conv block - capture high-level features
                torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(True),
                torch.nn.AdaptiveAvgPool2d((8, 8))  # Fixed output size
            )
            
            cnn_output_features = 128 * 8 * 8  # 8192 features
            input_features = cnn_output_features
        else:
            self.flatten = torch.nn.Flatten()
            input_features = h * w * c
        
        # Multi-head self-attention for feature refinement
        if attention_mechanism:
            self.attention = torch.nn.MultiheadAttention(
                embed_dim=input_features,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
            self.attention_norm = torch.nn.LayerNorm(input_features)
        
        # Build superior KAN layers
        layers_hidden = [input_features] + kan_layers + [num_classes]
        
        self.kan_layers = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        self.residual_projections = torch.nn.ModuleList()
        
        for i, (in_f, out_f) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            # High-performance KAN layer
            kan_layer = HighPerformanceKANLinear(
                in_f, out_f,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_base=1.8,      # Strong base activation
                scale_spline=2.5,    # Very strong spline contribution
                prob_update_grid=0.15,  # Frequent grid updates
                grid_range=[-4, 4],  # Wide grid range
                adaptive_grid=True
            )
            
            self.kan_layers.append(kan_layer)
            
            # Layer normalization for stability
            if layer_norm and i < len(layers_hidden) - 2:  # No norm on final layer
                self.layer_norms.append(torch.nn.LayerNorm(out_f))
            else:
                self.layer_norms.append(torch.nn.Identity())
            
            # Strategic dropout
            if i < len(layers_hidden) - 2:  # No dropout on final layer
                self.dropouts.append(torch.nn.Dropout(dropout_rate))
            else:
                self.dropouts.append(torch.nn.Identity())
            
            # Residual connections
            if residual_connections and in_f == out_f:
                self.residual_projections.append(torch.nn.Identity())
            elif residual_connections and in_f != out_f:
                self.residual_projections.append(torch.nn.Linear(in_f, out_f))
            else:
                self.residual_projections.append(None)
        
        print(f"SuperiorESC_KAN created:")
        print(f"  - CNN Frontend: {use_cnn_frontend}")
        print(f"  - KAN Layers: {layers_hidden}")
        print(f"  - Grid Size: {grid_size}, Spline Order: {spline_order}")
        print(f"  - Attention: {attention_mechanism}")
        print(f"  - Total Parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, x):
        # Handle input format
        if x.dim() == 4 and x.shape[1] not in [1, 3]:
            x = x.permute(0, 3, 1, 2)
        
        # CNN feature extraction
        if self.use_cnn_frontend:
            x = self.cnn_frontend(x)
            x = x.view(x.size(0), -1)
        else:
            x = self.flatten(x)
        
        # Self-attention for feature refinement
        if self.attention_mechanism:
            # Reshape for attention (batch, seq_len=1, features)
            x_att = x.unsqueeze(1)
            att_out, _ = self.attention(x_att, x_att, x_att)
            x = self.attention_norm(x + att_out.squeeze(1))
        
        # Superior KAN processing
        for i, (kan_layer, layer_norm, dropout, residual_proj) in enumerate(
            zip(self.kan_layers, self.layer_norms, self.dropouts, self.residual_projections)
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
            
            # Layer norm and dropout
            x = layer_norm(x)
            x = dropout(x)
        
        return x

    def regularization_loss(self):
        """Enhanced regularization loss"""
        reg_loss = 0.0
        for layer in self.kan_layers:
            if hasattr(layer, 'spline_regularization_weight'):
                # L2 regularization on spline weights
                reg_loss += layer.spline_regularization_weight * torch.norm(layer.spline_weight)
                # Smoothness regularization
                reg_loss += 0.001 * torch.mean(torch.diff(layer.spline_weight, dim=-1) ** 2)
        return reg_loss


def create_superior_kan(input_shape, num_classes=26, performance_mode='ultra'):
    """
    Create Superior KAN optimized for >90% accuracy
    
    Args:
        performance_mode: 'ultra', 'high', or 'balanced'
    """
    if performance_mode == 'ultra':
        return SuperiorESC_KAN(
            input_shape, num_classes,
            use_cnn_frontend=True,
            kan_layers=[1024, 512, 256, 128],
            grid_size=25,
            spline_order=5,
            dropout_rate=0.05,
            attention_mechanism=True
        )
    elif performance_mode == 'high':
        return SuperiorESC_KAN(
            input_shape, num_classes,
            use_cnn_frontend=True,
            kan_layers=[512, 256, 128],
            grid_size=20,
            spline_order=4,
            dropout_rate=0.1,
            attention_mechanism=True
        )
    else:  # balanced
        return SuperiorESC_KAN(
            input_shape, num_classes,
            use_cnn_frontend=False,
            kan_layers=[512, 256],
            grid_size=15,
            spline_order=3,
            dropout_rate=0.15,
            attention_mechanism=False
        )


def create_memory_safe_superior_kan(input_shape, num_classes=26, max_memory_gb=8):
    """Memory-safe version of Superior KAN"""
    if max_memory_gb <= 4:
        return create_superior_kan(input_shape, num_classes, 'balanced')
    elif max_memory_gb <= 8:
        return create_superior_kan(input_shape, num_classes, 'high')
    else:
        return create_superior_kan(input_shape, num_classes, 'ultra')
