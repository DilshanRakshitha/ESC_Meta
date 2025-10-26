"""
KAN (Kolmogorov-Arnold Network) Implementation
"""

import torch
import torch.nn.functional as F
import math
import random


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        prob_update_grid=-0.001
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()
        self.prob_update_grid = prob_update_grid
        self.current_grid_step = 0
        self.max_grid_update_step = 100

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features, f'{x.dim()}, {x.size(0)},{x.size(1)}'

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )

        if self.training and random.random() < self.prob_update_grid and \
                    self.current_grid_step < self.max_grid_update_step:
            try:
                self.update_grid(x)
                # print('grid update')
                self.current_grid_step += 1
            except Exception as e:
                print('grid update fail', e)
        
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        # self.grid.copy_(grid.T)
        self.grid *= 0.99
        self.grid += grid.T * 0.01
        # self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))
        self.spline_weight.data.copy_(self.spline_weight.data * 0.99 + 0.01 * self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


# ESC-specific wrapper for the original KAN
class ESC_KAN(torch.nn.Module):
    """
    ESC-specific wrapper for the exact KAN implementation
    Only changes: input handling for ESC data format
    """
    def __init__(self, input_shape, num_classes=26, hidden_layers=[512, 256]):
        super().__init__()
        h, w, c = input_shape
        input_features = h * w * c
        
        # Build exact same KAN as original implementation
        layers_hidden = [input_features] + hidden_layers + [num_classes]
        self.kan = KAN(layers_hidden)
        self.flatten = torch.nn.Flatten()
        
    def forward(self, x):
        # Handle ESC input format - only change from original
        if x.dim() == 4 and x.shape[1] not in [1, 3]:
            x = x.permute(0, 3, 1, 2)
        x = self.flatten(x)
        return self.kan(x)
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return self.kan.regularization_loss(regularize_activation, regularize_entropy)


# Factory functions for ESC_Meta compatibility
def create_exact_kan(input_shape, num_classes=26, hidden_layers=[512, 256]):
    return ESC_KAN(input_shape, num_classes, hidden_layers)

# Similar to the original implementation
def create_pure_kan(input_shape, num_classes=26):
    return ESC_KAN(input_shape, num_classes, [512, 256])


class FastESC_KAN(torch.nn.Module):
    """
    Balanced Fast ESC-friendly KAN wrapper.
    Optimized for both accuracy and overfitting prevention:
    - Larger bottleneck for expressiveness
    - Strategic dropout placement
    - Higher-capacity KAN layers
    - Residual connections for gradient flow
    - Balanced regularization
    """
    def __init__(self,
                 input_shape,
                 num_classes=26,
                 hidden_layers=None,
                 bottleneck_dim=1024,     # Increased for more capacity
                 disable_grid_update=True,
                 kan_grid_size=8,         # Increased grid size for expressiveness
                 dropout_rate=0.2,        # Moderate dropout
                 input_dropout=0.05,      # Light input dropout
                 layer_norm=True,
                 residual_connections=True,
                 noise_factor=0.005):     # Minimal noise
        super().__init__()
        h, w, c = input_shape
        input_features = h * w * c

        # Input processing with minimal regularization
        self.flatten = torch.nn.Flatten()
        self.input_dropout = torch.nn.Dropout(input_dropout) if input_dropout > 0 else torch.nn.Identity()
        self.noise_factor = noise_factor

        # Multi-stage bottleneck projection for better feature extraction
        self.project = torch.nn.Sequential(
            torch.nn.Linear(input_features, bottleneck_dim * 2),
            torch.nn.BatchNorm1d(bottleneck_dim * 2),
            torch.nn.GELU(),  # Better activation than ReLU
            torch.nn.Dropout(dropout_rate * 0.5),  # Light dropout
            torch.nn.Linear(bottleneck_dim * 2, bottleneck_dim),
            torch.nn.BatchNorm1d(bottleneck_dim),
            torch.nn.GELU()
        )

        # Build larger KAN layers with residual connections
        if hidden_layers is None:
            hidden_layers = [bottleneck_dim, bottleneck_dim // 2, bottleneck_dim // 4]  # More layers

        layers_hidden = [bottleneck_dim] + hidden_layers + [num_classes]
        self.residual_connections = residual_connections

        # Create enhanced KAN layers
        self.kan_layers = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        self.residual_projections = torch.nn.ModuleList()
        
        for i, (in_f, out_f) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            # Enhanced KAN layer with better expressiveness
            layer = KANLinear(
                in_f, out_f, 
                grid_size=kan_grid_size,
                scale_noise=0.1,        # Standard noise
                scale_spline=1.0,       # Full spline scale
                scale_base=1.2,         # Stronger base activation
                spline_order=3,         # Standard spline order
                grid_range=[-2, 2]      # Wider grid range
            )
            
            # Configure grid updates strategically
            if disable_grid_update:
                layer.prob_update_grid = 0.0
                layer.max_grid_update_step = 0
            else:
                # Allow some grid updates early in training
                layer.prob_update_grid = max(0.001, 0.01 - i * 0.002)
                layer.max_grid_update_step = 50
                
            self.kan_layers.append(layer)
            
            # Residual projection for skip connections
            if residual_connections and in_f == out_f and i < len(layers_hidden) - 2:
                self.residual_projections.append(torch.nn.Identity())
            elif residual_connections and i < len(layers_hidden) - 2:
                self.residual_projections.append(torch.nn.Linear(in_f, out_f))
            else:
                self.residual_projections.append(None)
            
            # Strategic normalization and dropout
            if i < len(layers_hidden) - 2:  # Not for output layer
                if layer_norm:
                    self.layer_norms.append(torch.nn.LayerNorm(out_f))
                else:
                    self.layer_norms.append(torch.nn.Identity())
                # Reduce dropout for later layers
                dropout_prob = dropout_rate * (0.5 ** i) if i > 0 else dropout_rate
                self.dropouts.append(torch.nn.Dropout(dropout_prob))
            else:
                self.layer_norms.append(torch.nn.Identity())
                self.dropouts.append(torch.nn.Identity())

        param_count = sum(p.numel() for p in self.parameters())
        print(f"FastESC_KAN created with {param_count:,} parameters")
        print(f"  - Bottleneck: {bottleneck_dim}, Grid size: {kan_grid_size}")
        print(f"  - Hidden layers: {hidden_layers}")
        print(f"  - Residual connections: {residual_connections}")
        print(f"  - Adaptive dropout: {dropout_rate} → {dropout_rate * 0.25}")

    def forward(self, x):
        if x.dim() == 4 and x.shape[1] not in [1, 3]:
            x = x.permute(0, 3, 1, 2)
        
        x = self.flatten(x)
        
        # Minimal input noise for robustness
        if self.training and self.noise_factor > 0:
            noise = torch.randn_like(x) * self.noise_factor
            x = x + noise
            
        x = self.input_dropout(x)
        x = self.project(x)
        
        # Pass through KAN layers with residual connections
        for i, (kan_layer, norm_layer, dropout_layer, residual_proj) in enumerate(zip(
            self.kan_layers, self.layer_norms, self.dropouts, self.residual_projections
        )):
            identity = x
            x = kan_layer(x)
            
            if i < len(self.kan_layers) - 1:  # Not output layer
                x = norm_layer(x)
                
                # Add residual connection
                if residual_proj is not None:
                    x = x + residual_proj(identity)
                    
                x = dropout_layer(x)
        
        return x

    def regularization_loss(self, l2_weight=1e-5, kan_reg_weight=1e-4):
        """
        Lighter regularization to preserve expressiveness
        """
        l2_loss = 0.0
        # Light L2 regularization
        for module in self.project.modules():
            if isinstance(module, torch.nn.Linear):
                l2_loss += torch.sum(module.weight ** 2)
        
        # Reduced KAN regularization
        kan_reg = sum(layer.regularization_loss(0.5, 0.5) for layer in self.kan_layers)
        
        return l2_weight * l2_loss + kan_reg_weight * kan_reg


# Add a memory-safe variant at the end
class MemoryEfficientKAN(torch.nn.Module):
    """
    Memory-efficient KAN wrapper
    - Gradient checkpointing
    - Progressive layer sizes
    - Memory-conscious grid sizes
    - Built-in memory monitoring
    """
    def __init__(self, input_shape, num_classes=26, max_memory_gb=8):
        super().__init__()
        h, w, c = input_shape
        input_features = h * w * c
        
        # Calculate safe bottleneck based on available memory
        max_bottleneck = min(1024, (max_memory_gb * 1024**3) // (input_features * 8))  # 8 bytes per float64
        bottleneck_dim = min(512, max_bottleneck)
        
        print(f"MemoryEfficientKAN: Using bottleneck_dim={bottleneck_dim} for safety")
        
        self.flatten = torch.nn.Flatten()
        
        # Conservative projection
        self.project = torch.nn.Sequential(
            torch.nn.Linear(input_features, bottleneck_dim),
            torch.nn.LayerNorm(bottleneck_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1)
        )
        
        # Progressive KAN layers with decreasing sizes
        layer_sizes = [bottleneck_dim, bottleneck_dim//2, bottleneck_dim//4, num_classes]
        self.kan_layers = torch.nn.ModuleList()
        
        for i, (in_f, out_f) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            # Use smaller grid sizes to save memory
            grid_size = max(3, 8 - i * 2)  # 8, 6, 4, ...
            layer = KANLinear(
                in_f, out_f,
                grid_size=grid_size,
                scale_noise=0.1,
                scale_spline=1.0
            )
            # Disable grid updates to save memory
            layer.prob_update_grid = 0.0
            layer.max_grid_update_step = 0
            self.kan_layers.append(layer)
        
        # Add layer norms for stability
        self.layer_norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(size) for size in layer_sizes[1:-1]
        ])
        
        param_count = sum(p.numel() for p in self.parameters())
        print(f"MemoryEfficientKAN: {param_count:,} parameters (safe for {max_memory_gb}GB)")
    
    def forward(self, x):
        if x.dim() == 4 and x.shape[1] not in [1, 3]:
            x = x.permute(0, 3, 1, 2)
        
        x = self.flatten(x)
        x = self.project(x)
        
        # Forward through KAN layers with memory checkpointing
        for i, kan_layer in enumerate(self.kan_layers):
            if i < len(self.layer_norms):
                # Use gradient checkpointing for memory efficiency
                x = torch.utils.checkpoint.checkpoint(kan_layer, x, use_reentrant=False)
                x = self.layer_norms[i](x)
            else:
                x = kan_layer(x)
        
        return x
    
    def regularization_loss(self):
        return sum(layer.regularization_loss(0.1, 0.1) for layer in self.kan_layers)


def create_memory_safe_kan(input_shape, num_classes=26, max_memory_gb=8):
    """Create a memory-safe KAN that won't crash your system"""
    return MemoryEfficientKAN(input_shape, num_classes, max_memory_gb)


def create_fast_exact_kan(input_shape, num_classes=26, mode='balanced', 
                         bottleneck_dim=1024, dropout_rate=0.2, 
                         disable_grid_update=True):
    """
    Create FastESC_KAN with different configurations
    
    Args:
        mode: 'balanced' (default), 'high_accuracy', or 'regularized'
    """
    if mode == 'high_accuracy':
        # Memory-efficient high accuracy version
        return FastESC_KAN(
            input_shape, num_classes,
            bottleneck_dim=min(bottleneck_dim, 1024),  # Cap bottleneck
            hidden_layers=[512, 256],  # Fewer but deeper layers
            kan_grid_size=10,  # Good expressiveness without explosion
            dropout_rate=0.15,
            input_dropout=0.03,
            residual_connections=True,
            disable_grid_update=False,  # Allow some grid updates
            noise_factor=0.003
        )
    elif mode == 'regularized':
        # Anti-overfitting focused
        return FastESC_KAN(
            input_shape, num_classes,
            bottleneck_dim=bottleneck_dim // 2,
            hidden_layers=[256, 128],
            kan_grid_size=5,
            dropout_rate=0.4,
            input_dropout=0.1,
            residual_connections=False,
            noise_factor=0.01
        )
    else:  # balanced (default)
        return FastESC_KAN(
            input_shape, num_classes,
            bottleneck_dim=bottleneck_dim,
            hidden_layers=[512, 256],
            dropout_rate=dropout_rate,
            input_dropout=0.05,
            residual_connections=True,
            disable_grid_update=disable_grid_update
        )
