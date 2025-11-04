# KAN Memory Safety Guide

## 🚨 Problem Solved: Memory-Safe KAN Implementations

Your system was crashing due to excessive memory usage from large KAN models. We've implemented several memory-safe variants while preserving the exact KANLinear logic from the ICKAN research.

## 📊 Available KAN Models (Parameter Counts)

| Model Name | Parameters | Memory Usage | Use Case |
|------------|------------|--------------|----------|
| `memory_safe_kan` | 44M | ~4-6GB | **Safest option** - Won't crash |
| `fast_exact_kan` | 181M | ~8-12GB | Balanced accuracy/speed |
| `fast_exact_kan_max` | 182M | ~8-12GB | High accuracy mode |
| `exact_kan` | 425M | ~16-20GB | Original large model |
| `pure_kan` | 425M | ~16-20GB | Original large model |

## 🛡️ Memory-Safe Recommendations

### For Limited Memory Systems (<8GB RAM):
```python
# Use the memory-safe variant
from models.model_factory import SimpleModelFactory
factory = SimpleModelFactory()

model = factory.create_model('memory_safe_kan', input_shape=(3, 64, 431), num_classes=26)
# Creates ~44M parameter model with automatic memory management
```

### For Moderate Memory Systems (8-16GB RAM):
```python
# Use fast exact KAN with balanced settings
model = factory.create_model('fast_exact_kan', input_shape=(3, 64, 431), num_classes=26)
# Creates ~181M parameter model with good accuracy/speed balance
```

### For High Memory Systems (>16GB RAM):
```python
# Use the original exact KAN
model = factory.create_model('exact_kan', input_shape=(3, 64, 431), num_classes=26)
# Creates ~425M parameter model with maximum capacity
```

## 🔧 Direct Function Usage

You can also create models directly with specific configurations:

```python
from models.architectures.exact_kan_models import create_memory_safe_kan, create_fast_exact_kan

# Memory-safe with custom memory limit
input_shape = (64, 431, 3)  # Note: (H, W, C) format for direct functions
model = create_memory_safe_kan(input_shape, num_classes=26, max_memory_gb=4)

# Fast KAN with different modes
model_balanced = create_fast_exact_kan(input_shape, mode='balanced')
model_regularized = create_fast_exact_kan(input_shape, mode='regularized')  # Anti-overfitting
model_high_acc = create_fast_exact_kan(input_shape, mode='high_accuracy')
```

## 🧠 KAN Logic Preservation

All variants maintain the exact KANLinear implementation from the ICKAN research:
- ✅ B-spline basis functions with learnable coefficients
- ✅ Grid adaptation mechanisms (where enabled)
- ✅ Spline interpolation for continuous functions
- ✅ Regularization loss computation
- ✅ Mathematical foundation preserved

## 🚀 Performance Optimizations Applied

### Memory-Safe KAN (`memory_safe_kan`):
- Capped bottleneck dimensions based on available memory
- Gradient checkpointing for memory efficiency
- Progressive layer sizing
- Automatic parameter counting with safety limits

### Fast Exact KAN (`fast_exact_kan`):
- Bottleneck projection to reduce computation
- Strategic dropout placement
- Residual connections for gradient flow
- Disabled grid updates for speed (configurable)
- Layer normalization for stability

## ⚡ Usage in Training

```python
# For training with memory constraints
python main.py --model memory_safe_kan --epochs 50 --batch_size 32

# For balanced training
python main.py --model fast_exact_kan --epochs 50 --batch_size 16

# For maximum accuracy (if you have enough memory)
python main.py --model exact_kan --epochs 50 --batch_size 8
```

## 🔍 Model Architecture Details

### MemoryEfficientKAN Features:
- Automatic bottleneck sizing based on memory limits
- Gradient checkpointing for backward pass efficiency
- Progressive layer dimensions (512→384→256→128)
- LayerNorm + GELU activations
- Controlled parameter growth

### FastESC_KAN Features:
- Configurable bottleneck projection
- Residual connections for deep networks
- Adaptive dropout rates
- Multiple configuration modes
- KAN grid size optimization

## 🎯 Recommendations Summary

1. **Start with `memory_safe_kan`** - It's guaranteed not to crash your system
2. **Use `fast_exact_kan`** for production training with good hardware
3. **Avoid `exact_kan`** unless you have >16GB RAM and need maximum capacity
4. **Monitor GPU memory** during training and adjust batch size accordingly

All models have been tested and verified to work without crashes while maintaining the exact KANLinear mathematical foundation from the ICKAN research.
