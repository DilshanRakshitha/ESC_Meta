# Hyperparameter Optimization System

## Overview
This module provides a modular and configurable hyperparameter optimization system using Optuna for your ESC Meta models.

## Features
- **Configurable optimization**: Easy-to-modify YAML configurations
- **Multiple optimization modes**: Quick testing, standard, and extensive tuning
- **Model-agnostic**: Works with AlexNet, KAN, ICKAN, and WavKAN
- **Pruning support**: Early stopping of unpromising trials
- **Visualization**: Automatic generation of optimization plots
- **Persistence**: Save and load optimization studies

## Quick Start

### 1. Basic Usage
```bash
# Quick test (5 trials, 2 CV folds, 10 epochs)
python optimize_models.py --model alexnet --config quick

# Standard optimization (20 trials, 3 CV folds, 30 epochs)
python optimize_models.py --model kan --config standard

# Extensive optimization (100 trials, 5 CV folds, 50 epochs)
python optimize_models.py --model all --config extensive
```

### 2. Programmatic Usage
```python
from optimization import HyperparameterOptimizer
from main import FSCMetaMain

# Load data
pipeline = FSCMetaMain()
features, labels = pipeline.load_data('auto')

# Create optimizer
optimizer = HyperparameterOptimizer()

# Quick test mode
optimizer.quick_test_mode()

# Define model factory
def alexnet_factory(input_shape, num_classes, **kwargs):
    from models.architectures.AlexNet import AlexNet
    return AlexNet(input_size=input_shape[0], num_classes=num_classes)

# Run optimization
results = optimizer.optimize(alexnet_factory, (features, labels), 'alexnet')
```

## Configuration

### Configuration Files
- `config/optimization_configs.yml`: Contains quick, standard, and extensive configurations

### Configuration Options

#### Study Settings
```yaml
study:
  study_name: 'esc_meta_optimization'
  n_trials: 20          # Number of trials to run
  timeout: 3600         # Maximum time in seconds
  direction: 'maximize' # Optimize for maximum accuracy
```

#### Training Settings
```yaml
training:
  max_epochs: 30        # Maximum training epochs
  patience: 8           # Early stopping patience
  batch_size_options: [32, 64]  # Batch size choices
```

#### Hyperparameter Ranges
```yaml
hyperparameters:
  common:
    learning_rate:
      type: 'loguniform'
      low: 1e-5
      high: 1e-2
    optimizer:
      type: 'categorical'
      choices: ['adam', 'adamw']
```

## Optimization Modes

### Quick Test Mode
- **Purpose**: Fast testing and debugging
- **Trials**: 5
- **CV Folds**: 2
- **Epochs**: 10
- **Time**: ~5 minutes

### Standard Mode (Default)
- **Purpose**: Balanced optimization
- **Trials**: 20
- **CV Folds**: 3
- **Epochs**: 30
- **Time**: ~1 hour

### Extensive Mode
- **Purpose**: Thorough optimization
- **Trials**: 100
- **CV Folds**: 5
- **Epochs**: 50
- **Time**: ~2+ hours

## Output Structure

```
optimization/
├── logs/
│   ├── optimization.log
│   ├── alexnet_best_params.yaml
│   ├── alexnet_optimization_results.pkl
│   └── alexnet_trials.csv
├── plots/
│   ├── alexnet_optimization_history.png
│   ├── alexnet_param_importances.png
│   └── alexnet_parallel_coordinate.png
└── optuna_study.db  # SQLite database for studies
```

## Model-Specific Hyperparameters

### AlexNet
- `dropout_rate`: Dropout probability (0.3-0.7)

### KAN/ICKAN/WavKAN
- `hidden_dim`: Hidden layer dimension (256, 512, 1024)
- `dropout_rate`: Dropout probability (0.1-0.5)

### Common Parameters
- `learning_rate`: Learning rate (1e-5 to 1e-2)
- `batch_size`: Batch size (16, 32, 64, 128)
- `optimizer`: Optimizer choice (adam, adamw, sgd)
- `weight_decay`: L2 regularization (1e-7 to 1e-2)

## Examples

### Example 1: Quick Test All Models
```bash
python optimize_models.py --model all --config quick
```

### Example 2: Extensive KAN Optimization
```bash
python optimize_models.py --model kan --config extensive
```

### Example 3: Custom Configuration
```python
from optimization import OptimizationConfig, HyperparameterOptimizer

# Create custom config
config = OptimizationConfig()
config.config['study']['n_trials'] = 50
config.config['training']['max_epochs'] = 25

optimizer = HyperparameterOptimizer()
optimizer.config = config

# Run optimization...
```

## Best Practices

1. **Start with Quick Mode**: Always test with quick mode first
2. **Monitor Progress**: Check logs and plots regularly
3. **Use Pruning**: Enable pruning for faster convergence
4. **Resource Management**: Extensive mode requires significant compute
5. **Save Results**: Always save best parameters for reproduction

## Troubleshooting

### Common Issues

1. **GPU Memory Error**: Set `device: 'cpu'` in configuration
2. **Import Errors**: Ensure all dependencies are installed
3. **Long Runtime**: Use smaller n_trials or quick mode
4. **No Improvement**: Check hyperparameter ranges

### Performance Tips

1. **Reduce CV folds** for faster optimization
2. **Use pruning** to stop poor trials early
3. **Limit batch size options** for memory constraints
4. **Run on CPU** if GPU memory is limited

## Advanced Usage

### Custom Model Factory
```python
def custom_model_factory(input_shape, num_classes, **kwargs):
    # Your custom model creation logic
    hidden_dim = kwargs.get('hidden_dim', 512)
    dropout = kwargs.get('dropout_rate', 0.3)
    
    model = YourCustomModel(
        input_shape=input_shape,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout
    )
    return model

# Use with optimizer
results = optimizer.optimize(custom_model_factory, data, 'custom_model')
```

### Resume Optimization
```python
# Load existing study
optimizer.load_study('esc_meta_optimization_alexnet')

# Continue optimization
results = optimizer.optimize(model_factory, data, 'alexnet')
```

## Results Analysis

The optimization system provides:
- **Best parameters**: Optimal hyperparameters found
- **Performance metrics**: Cross-validation scores and statistics
- **Convergence plots**: Optimization history visualization
- **Parameter importance**: Which parameters matter most
- **Trial database**: Complete record of all trials

Use these results to:
1. Reproduce best models
2. Understand parameter sensitivity
3. Guide future optimization efforts
4. Compare different models
