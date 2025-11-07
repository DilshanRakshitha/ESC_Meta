import sys
import argparse
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from main import load_fsc22_folds
from optimization.hyperparameter_optimizer import HyperparameterOptimizer
from optimization.optimization_config import OptimizationConfig

from models.architectures.AlexNet import AlexNet
try:
    from models.architectures.KAN_inspired_models import create_high_performance_kan_inspired
except ImportError:
    print("Warning: Could not import KAN models")
    create_high_performance_kan_inspired = None
    
try:
    from models.architectures.ICKAN_inspired_models import create_high_performance_ickan_inspired_model
except ImportError:
    print("Warning: Could not import ICKAN models")
    create_high_performance_ickan_inspired_model = None
    
try:
    from models.architectures.WavKAN_inspired_models import create_high_performance_wavkan_inspired_model
except ImportError:
    print("Warning: Could not import WavKAN models")
    create_high_performance_wavkan_inspired_model = None

def create_alexnet_factory():
    def factory(input_shape, num_classes, **kwargs):
        input_channels = input_shape[0] if len(input_shape) == 3 else 3
        dropout_rate = kwargs.get('dropout_rate', 0.5)
        model = AlexNet(input_size=input_channels, num_classes=num_classes)
        if hasattr(model, 'Dropout_1'):
            model.Dropout_1.p = dropout_rate
        if hasattr(model, 'Dropout_2'):
            model.Dropout_2.p = dropout_rate
        return model
    factory.__name__ = 'create_alexnet'
    return factory

def create_kan_factory():
    if create_high_performance_kan_inspired is None:
        raise ValueError("KAN models not available")
    def factory(input_shape, num_classes, **kwargs):
        if len(input_shape) == 3:
            kan_input_shape = (input_shape[1], input_shape[2], input_shape[0])
        else:
            kan_input_shape = input_shape
        return create_high_performance_kan_inspired(kan_input_shape, num_classes)
    factory.__name__ = 'create_kan'
    return factory

def create_ickan_factory():
    if create_high_performance_ickan_inspired_model is None:
        raise ValueError("ICKAN models not available")
    def factory(input_shape, num_classes, **kwargs):
        if len(input_shape) == 3:
            ickan_input_shape = (input_shape[1], input_shape[2], input_shape[0])
        else:
            ickan_input_shape = input_shape
        return create_high_performance_ickan_inspired_model(ickan_input_shape, num_classes)
    factory.__name__ = 'create_ickan'
    return factory

def create_wavkan_factory():
    if create_high_performance_wavkan_inspired_model is None:
        raise ValueError("WavKAN models not available")
    def factory(input_shape, num_classes, **kwargs):
        if len(input_shape) == 3:
            wavkan_input_shape = (input_shape[1], input_shape[2], input_shape[0])
        else:
            wavkan_input_shape = input_shape
        return create_high_performance_wavkan_inspired_model(wavkan_input_shape, num_classes)
    factory.__name__ = 'create_wavkan'
    return factory

def load_data(model_name='alexnet'):
    print("Loading FSC22 data for optimization...")
    result = load_fsc22_folds()
    if not result:
        raise ValueError("Failed to load FSC22 data")
    
    folds_data = result['folds_data']
    all_features = []
    all_labels = []
    
    for fold_name, fold_data in folds_data.items():
        for feature, label in fold_data:
            all_features.append(feature)
            all_labels.append(label)
    
    features = np.array(all_features)
    labels = np.array(all_labels)
    
    # Apply same dimension transformation as in training
    is_kan_model = 'kan' in model_name.lower()
    if len(features.shape) == 4 and features.shape[-1] == 3 and not is_kan_model:
        # CNN models expect (batch, channels, height, width) format
        features = np.transpose(features, (0, 3, 1, 2))  # (B,H,W,C) -> (B,C,H,W)
        print(f"Transposed features for CNN model: {features.shape}")
    
    print(f"Data loaded: {len(features)} samples, {len(np.unique(labels))} classes")
    print(f"Feature shape: {features.shape}")
    return features, labels

def optimize_model(model_name: str, config_type: str = 'standard'):
    print(f"\nStarting hyperparameter optimization for {model_name.upper()}")
    print(f"Configuration: {config_type}")
    print("=" * 70)
    features, labels = load_data(model_name)
    
    factories = {}
    factories['alexnet'] = create_alexnet_factory()
    
    try:
        factories['kan'] = create_kan_factory()
    except ValueError as e:
        print(f"KAN not available: {e}")
    
    try:
        factories['ickan'] = create_ickan_factory()
    except ValueError as e:
        print(f"ICKAN not available: {e}")
        
    try:
        factories['wavkan'] = create_wavkan_factory()
    except ValueError as e:
        print(f"WavKAN not available: {e}")
    
    if model_name not in factories:
        available_models = list(factories.keys())
        raise ValueError(f"Model '{model_name}' not available. Choose from: {available_models}")
    
    model_factory = factories[model_name]
    config_path = "config/optimization_configs.yml"
    optimizer = HyperparameterOptimizer(config_path)
    if config_type == 'quick':
        optimizer.quick_test_mode()
    elif config_type == 'extensive':
        optimizer.extensive_tuning_mode()
    results = optimizer.optimize(model_factory, (features, labels), model_name)
    if 'error' not in results:
        print(f"\nOPTIMIZATION COMPLETED FOR {model_name.upper()}")
        print("=" * 70)
        print(f"Best accuracy: {results['best_score']:.4f} ± {results['cv_std']:.4f}")
        print(f"Best parameters:")
        for param, value in results['best_params'].items():
            print(f"  {param}: {value}")
        print(f"\nResults saved in: optimization/logs/")
        print(f"Plots saved in: optimization/plots/")
    else:
        print(f"Optimization failed: {results['error']}")
    return results

def optimize_all_models(config_type: str = 'quick'):
    models = ['alexnet', 'kan', 'ickan', 'wavkan']
    results = {}
    print(f"\nOPTIMIZING ALL MODELS ({config_type.upper()} MODE)")
    print("=" * 80)
    for model_name in models:
        try:
            results[model_name] = optimize_model(model_name, config_type)
        except Exception as e:
            print(f"Failed to optimize {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    print(f"\nOPTIMIZATION SUMMARY")
    print("=" * 80)
    for model_name, result in results.items():
        if 'error' not in result:
            print(f"{model_name.upper():>10}: {result['best_score']:.4f} ± {result['cv_std']:.4f}")
        else:
            print(f"{model_name.upper():>10}: FAILED")
    return results

def main():
    parser = argparse.ArgumentParser(description='ESC Meta Hyperparameter Optimization')
    parser.add_argument('--model', type=str, choices=['alexnet', 'kan', 'ickan', 'wavkan', 'all'],
                       default='alexnet', help='Model to optimize')
    parser.add_argument('--config', type=str, choices=['quick', 'standard', 'extensive'],
                       default='quick', help='Optimization configuration')
    args = parser.parse_args()
    print("ESC Meta - Hyperparameter Optimization")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    if args.model == 'all':
        results = optimize_all_models(args.config)
    else:
        results = optimize_model(args.model, args.config)
    print("\nOptimization pipeline completed!")
    return results

if __name__ == "__main__":
    results = main()
