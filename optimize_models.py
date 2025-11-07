import sys
import argparse
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from main import FSCMetaMain
from optimization.hyperparameter_optimizer import HyperparameterOptimizer
from optimization.optimization_config import OptimizationConfig

from models.architectures.AlexNet import AlexNet
from models.architectures.KAN_inspired_models import create_high_performance_kan_inspired
from models.architectures.ickan_models import create_high_performance_ickan
from models.architectures.wavkan_models import create_high_performance_wavkan

def create_alexnet_factory():
    """Factory function for AlexNet"""
    def factory(input_shape, num_classes, **kwargs):
        input_channels = input_shape[0] if len(input_shape) == 3 else 3
        dropout_rate = kwargs.get('dropout_rate', 0.5)
        
        # Create AlexNet and modify dropout if needed
        model = AlexNet(input_size=input_channels, num_classes=num_classes)
        
        # Update dropout rates if specified
        if hasattr(model, 'Dropout_1'):
            model.Dropout_1.p = dropout_rate
        if hasattr(model, 'Dropout_2'):
            model.Dropout_2.p = dropout_rate
            
        return model
    
    factory.__name__ = 'create_alexnet'
    return factory

def create_kan_factory():
    """Factory function for KAN"""
    def factory(input_shape, num_classes, **kwargs):
        # Convert (C, H, W) to (H, W, C) for KAN
        if len(input_shape) == 3:
            kan_input_shape = (input_shape[1], input_shape[2], input_shape[0])
        else:
            kan_input_shape = input_shape
        return create_high_performance_kan_inspired(kan_input_shape, num_classes)
    
    factory.__name__ = 'create_kan'
    return factory

def create_ickan_factory():
    """Factory function for ICKAN"""
    def factory(input_shape, num_classes, **kwargs):
        # Convert (C, H, W) to (H, W, C) for ICKAN
        if len(input_shape) == 3:
            ickan_input_shape = (input_shape[1], input_shape[2], input_shape[0])
        else:
            ickan_input_shape = input_shape
        return create_high_performance_ickan(ickan_input_shape, num_classes)
    
    factory.__name__ = 'create_ickan'
    return factory

def create_wavkan_factory():
    """Factory function for WavKAN"""
    def factory(input_shape, num_classes, **kwargs):
        # Convert (C, H, W) to (H, W, C) for WavKAN
        if len(input_shape) == 3:
            wavkan_input_shape = (input_shape[1], input_shape[2], input_shape[0])
        else:
            wavkan_input_shape = input_shape
        return create_high_performance_wavkan(wavkan_input_shape, num_classes)
    
    factory.__name__ = 'create_wavkan'
    return factory

def load_data():
    """Load data using existing FSCMetaMain pipeline"""
    print("📂 Loading data using FSCMetaMain pipeline...")
    
    pipeline = FSCMetaMain()
    features, labels = pipeline.load_data('auto')
    
    print(f"✅ Data loaded: {len(features)} samples, {len(np.unique(labels))} classes")
    print(f"📏 Feature shape: {features.shape}")
    
    return features, labels

def optimize_model(model_name: str, config_type: str = 'standard'):
    """
    Optimize hyperparameters for a specific model
    
    Args:
        model_name: 'alexnet', 'kan', 'ickan', or 'wavkan'
        config_type: 'quick', 'standard', or 'extensive'
    """
    print(f"\n🚀 Starting hyperparameter optimization for {model_name.upper()}")
    print(f"🔧 Configuration: {config_type}")
    print("=" * 70)
    
    # Load data
    features, labels = load_data()
    
    # Create model factory
    factories = {
        'alexnet': create_alexnet_factory(),
        'kan': create_kan_factory(),
        'ickan': create_ickan_factory(),
        'wavkan': create_wavkan_factory()
    }
    
    if model_name not in factories:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(factories.keys())}")
    
    model_factory = factories[model_name]
    
    # Load appropriate configuration
    config_path = "config/optimization_configs.yml"
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(config_path)
    
    # Switch to appropriate mode
    if config_type == 'quick':
        optimizer.quick_test_mode()
    elif config_type == 'extensive':
        optimizer.extensive_tuning_mode()
    
    # Run optimization
    results = optimizer.optimize(model_factory, (features, labels), model_name)
    
    # Print final results
    if 'error' not in results:
        print(f"\n🏆 OPTIMIZATION COMPLETED FOR {model_name.upper()}")
        print("=" * 70)
        print(f"Best accuracy: {results['best_score']:.4f} ± {results['cv_std']:.4f}")
        print(f"Best parameters:")
        for param, value in results['best_params'].items():
            print(f"  {param}: {value}")
        print(f"\nResults saved in: optimization/logs/")
        print(f"Plots saved in: optimization/plots/")
    else:
        print(f"❌ Optimization failed: {results['error']}")
    
    return results

def optimize_all_models(config_type: str = 'quick'):
    """Optimize all models"""
    models = ['alexnet', 'kan', 'ickan', 'wavkan']
    results = {}
    
    print(f"\n🔥 OPTIMIZING ALL MODELS ({config_type.upper()} MODE)")
    print("=" * 80)
    
    for model_name in models:
        try:
            results[model_name] = optimize_model(model_name, config_type)
        except Exception as e:
            print(f"❌ Failed to optimize {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    # Print summary
    print(f"\n📊 OPTIMIZATION SUMMARY")
    print("=" * 80)
    for model_name, result in results.items():
        if 'error' not in result:
            print(f"{model_name.upper():>10}: {result['best_score']:.4f} ± {result['cv_std']:.4f}")
        else:
            print(f"{model_name.upper():>10}: FAILED")
    
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='ESC Meta Hyperparameter Optimization')
    parser.add_argument('--model', type=str, choices=['alexnet', 'kan', 'ickan', 'wavkan', 'all'],
                       default='alexnet', help='Model to optimize')
    parser.add_argument('--config', type=str, choices=['quick', 'standard', 'extensive'],
                       default='quick', help='Optimization configuration')
    
    args = parser.parse_args()
    
    print("🎯 ESC Meta - Hyperparameter Optimization")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    
    if args.model == 'all':
        results = optimize_all_models(args.config)
    else:
        results = optimize_model(args.model, args.config)
    
    print("\n✅ Optimization pipeline completed!")
    
    return results

if __name__ == "__main__":
    results = main()
