"""
FSC Meta Optimization Interface
Main interface for hyperparameter optimization integrated with existing pipeline
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.extend([
    str(project_root),
    str(project_root / 'optimization'),
    str(project_root / 'models'),
    str(project_root / 'models' / 'architectures'),
])

from optimization.hyperparameter_tuner import OptunaTuner
from optimization.config import OptimizationConfig, OptimizationPresets, MODEL_STRATEGIES

class FSCMetaOptimizer:
    """
    Main optimization interface for FSC Meta models
    Integrates with existing pipeline architecture
    """
    
    def __init__(self, 
                 data_loader=None,
                 model_factory=None,
                 config_path: str = None):
        """
        Initialize FSC Meta Optimizer
        
        Args:
            data_loader: Existing data loader instance
            model_factory: Existing model factory instance
            config_path: Path to configuration file
        """
        # Import existing components
        try:
            if model_factory is None:
                # Import your existing SimpleModelFactory
                sys.path.append(str(project_root))
                from main import SimpleModelFactory
                self.model_factory = SimpleModelFactory()
            else:
                self.model_factory = model_factory
            
            if data_loader is None:
                # Import your existing data loader
                from features.fsc_original_features import FSCOriginalDataLoader
                self.data_loader = FSCOriginalDataLoader()
            else:
                self.data_loader = data_loader
                
        except ImportError as e:
            print(f"⚠️ Import warning: {e}")
            self.model_factory = model_factory
            self.data_loader = data_loader
        
        # Initialize optimization components
        self.optimization_config = OptimizationConfig()
        self.tuner = OptunaTuner(
            model_factory=self.model_factory,
            data_loader=self.data_loader,
            results_dir="optimization/results"
        )
        
        # Load existing configuration if available
        self.existing_config = self._load_existing_config(config_path)
        
        print("🔍 FSC Meta Optimizer initialized")
        print(f"   Model factory: {type(self.model_factory).__name__}")
        print(f"   Data loader: {type(self.data_loader).__name__}")
    
    def _load_existing_config(self, config_path: str) -> Dict[str, Any]:
        """Load existing configuration from main pipeline"""
        if config_path and os.path.exists(config_path):
            try:
                import yaml
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"⚠️ Could not load config: {e}")
        return {}
    
    def load_data(self, strategy: str = 'auto') -> tuple:
        """
        Load data using existing data loader
        """
        print("📂 Loading data for optimization...")
        
        try:
            if hasattr(self.data_loader, 'load_fsc_original_pickle_data'):
                features, labels = self.data_loader.load_fsc_original_pickle_data()
                print(f"✅ Loaded FSC Original data: {features.shape}")
                return features, labels
            else:
                # Fallback to synthetic data
                print("🎲 Using synthetic data for optimization testing...")
                n_samples, n_features = 1000, 128
                features = np.random.randn(n_samples, 3, 128, 196)  # Match expected shape
                labels = np.random.randint(0, 26, n_samples)
                return features, labels
                
        except Exception as e:
            print(f"⚠️ Data loading failed: {e}")
            # Generate synthetic data
            print("🎲 Generating synthetic data...")
            n_samples = 1000
            features = np.random.randn(n_samples, 3, 128, 196)
            labels = np.random.randint(0, 26, n_samples)
            return features, labels
    
    def optimize_single_model(self, 
                            model_name: str,
                            preset: str = 'development',
                            custom_trials: int = None,
                            custom_timeout: int = None) -> Dict[str, Any]:
        """
        Optimize a single model
        
        Args:
            model_name: Name of model to optimize ('alexnet', 'kan', 'ickan', 'wavkan')
            preset: Optimization preset ('quick_test', 'development', 'production', 'extensive')
            custom_trials: Custom number of trials (overrides preset)
            custom_timeout: Custom timeout in minutes (overrides preset)
        
        Returns:
            Optimization results dictionary
        """
        print(f"🎯 Optimizing {model_name.upper()}")
        
        # Load data
        features, labels = self.load_data()
        
        # Get preset configuration
        preset_config = getattr(OptimizationPresets, preset)()
        
        # Override with custom values if provided
        n_trials = custom_trials or preset_config['n_trials']
        timeout = (custom_timeout or preset_config['timeout_minutes']) * 60
        
        # Get model-specific suggestions
        strategy = MODEL_STRATEGIES.get(model_name.lower(), {})
        suggested_trials = strategy.get('suggested_trials', n_trials)
        
        print(f"📊 Optimization settings:")
        print(f"   Preset: {preset}")
        print(f"   Trials: {min(n_trials, suggested_trials)}")
        print(f"   Timeout: {timeout//60} minutes")
        print(f"   Strategy focus: {strategy.get('optimization_focus', 'general')}")
        
        # Run optimization
        results = self.tuner.optimize_model(
            model_name=model_name,
            features=features,
            labels=labels,
            n_trials=min(n_trials, suggested_trials),
            timeout=timeout
        )
        
        return results
    
    def optimize_all_models(self, 
                           models: List[str] = None,
                           preset: str = 'development',
                           parallel: bool = False) -> Dict[str, Dict]:
        """
        Optimize all models with intelligent scheduling
        
        Args:
            models: List of models to optimize (None for all)
            preset: Optimization preset
            parallel: Whether to run optimizations in parallel (experimental)
        
        Returns:
            Combined optimization results
        """
        if models is None:
            models = ['alexnet', 'kan', 'ickan', 'wavkan']
        
        print(f"🚀 COMPREHENSIVE OPTIMIZATION")
        print(f"   Models: {', '.join(models)}")
        print(f"   Preset: {preset}")
        print(f"   Parallel: {parallel}")
        
        # Load data once for all models
        features, labels = self.load_data()
        
        if parallel:
            return self._optimize_parallel(models, features, labels, preset)
        else:
            return self._optimize_sequential(models, features, labels, preset)
    
    def _optimize_sequential(self, models: List[str], features: np.ndarray, 
                           labels: np.ndarray, preset: str) -> Dict[str, Dict]:
        """Sequential optimization with intelligent ordering"""
        
        # Sort models by complexity (simpler models first for faster feedback)
        model_complexity = {'alexnet': 1, 'kan': 2, 'ickan': 3, 'wavkan': 4}
        sorted_models = sorted(models, key=lambda x: model_complexity.get(x, 999))
        
        results = {}
        for model_name in sorted_models:
            try:
                print(f"\n{'='*60}")
                print(f"🎯 OPTIMIZING {model_name.upper()}")
                print(f"{'='*60}")
                
                # Get preset and strategy
                preset_config = getattr(OptimizationPresets, preset)()
                strategy = MODEL_STRATEGIES.get(model_name.lower(), {})
                
                n_trials = strategy.get('suggested_trials', preset_config['n_trials'])
                timeout = preset_config['timeout_minutes'] * 60
                
                result = self.tuner.optimize_model(
                    model_name=model_name,
                    features=features,
                    labels=labels,
                    n_trials=n_trials,
                    timeout=timeout
                )
                
                results[model_name] = result
                
                # Generate visualization for this model
                try:
                    self.tuner.create_visualization(model_name)
                except Exception as e:
                    print(f"⚠️ Visualization failed for {model_name}: {e}")
                
            except Exception as e:
                print(f"❌ Optimization failed for {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        # Generate final report
        self._generate_comprehensive_report(results)
        
        return results
    
    def _optimize_parallel(self, models: List[str], features: np.ndarray, 
                          labels: np.ndarray, preset: str) -> Dict[str, Dict]:
        """Parallel optimization (experimental)"""
        print("🔄 Parallel optimization not fully implemented yet")
        print("   Falling back to sequential optimization...")
        return self._optimize_sequential(models, features, labels, preset)
    
    def _generate_comprehensive_report(self, results: Dict[str, Dict]):
        """Generate comprehensive optimization report"""
        
        report_file = Path("optimization/results/comprehensive_optimization_report.md")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write("# FSC Meta Hyperparameter Optimization Report\n\n")
            f.write(f"Generated on: {pd.Timestamp.now()}\n\n")
            
            # Summary table
            f.write("## Summary\n\n")
            f.write("| Model | Best Accuracy | Trials | Status |\n")
            f.write("|-------|---------------|--------|---------|\n")
            
            valid_results = []
            for model_name, result in results.items():
                if 'error' in result:
                    f.write(f"| {model_name.upper()} | - | - | ❌ Failed |\n")
                else:
                    f.write(f"| {model_name.upper()} | {result['best_value']:.4f} | {result['n_trials']} | ✅ Success |\n")
                    valid_results.append((model_name, result))
            
            f.write("\n")
            
            # Best model
            if valid_results:
                best_model, best_result = max(valid_results, key=lambda x: x[1]['best_value'])
                f.write(f"## Best Model: {best_model.upper()}\n\n")
                f.write(f"**Accuracy**: {best_result['best_value']:.4f}\n\n")
                f.write("**Best Parameters**:\n")
                for param, value in best_result['best_params'].items():
                    f.write(f"- {param}: {value}\n")
                f.write("\n")
            
            # Detailed results
            f.write("## Detailed Results\n\n")
            for model_name, result in results.items():
                f.write(f"### {model_name.upper()}\n\n")
                
                if 'error' in result:
                    f.write(f"❌ **Failed**: {result['error']}\n\n")
                    continue
                
                f.write(f"- **Best Accuracy**: {result['best_value']:.4f}\n")
                f.write(f"- **Trials Completed**: {result['n_trials']}\n")
                f.write(f"- **Optimization Focus**: {MODEL_STRATEGIES.get(model_name, {}).get('optimization_focus', 'general')}\n\n")
                
                f.write("**Optimized Parameters**:\n")
                for param, value in result['best_params'].items():
                    f.write(f"- {param}: {value}\n")
                f.write("\n")
        
        print(f"📊 Comprehensive report saved to: {report_file}")
    
    def load_optimized_model(self, model_name: str, input_shape: tuple, num_classes: int):
        """
        Load model with optimized hyperparameters
        """
        best_params = self.tuner.load_best_params(model_name)
        
        if best_params is None:
            print(f"⚠️ No optimized parameters found for {model_name}")
            print(f"   Run optimization first: optimize_single_model('{model_name}')")
            return None
        
        print(f"🎯 Loading {model_name} with optimized parameters")
        
        # Create model with optimized parameters
        model = self.tuner.create_optimized_model(model_name, best_params, input_shape, num_classes)
        
        if model is not None:
            print(f"✅ Optimized {model_name} loaded successfully")
            print(f"   Expected accuracy: {self.tuner.best_results.get(model_name, {}).get('best_value', 'Unknown')}")
        
        return model, best_params
    
    def compare_optimization_results(self) -> pd.DataFrame:
        """
        Compare optimization results across all models
        """
        results_dir = Path("optimization/results")
        
        if not results_dir.exists():
            print("⚠️ No optimization results found")
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_file in results_dir.glob("*_optimization_results.json"):
            try:
                import json
                with open(model_file, 'r') as f:
                    result = json.load(f)
                
                row = {
                    'Model': result['model_name'].upper(),
                    'Best_Accuracy': result['best_value'],
                    'Trials': result['n_trials'],
                    'Learning_Rate': result['best_params'].get('learning_rate', 'N/A'),
                    'Batch_Size': result['best_params'].get('batch_size', 'N/A'),
                    'Optimizer': result['best_params'].get('optimizer', 'N/A'),
                }
                
                comparison_data.append(row)
                
            except Exception as e:
                print(f"⚠️ Could not load {model_file}: {e}")
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            df = df.sort_values('Best_Accuracy', ascending=False)
            
            print("🏆 OPTIMIZATION RESULTS COMPARISON")
            print("="*60)
            print(df.to_string(index=False))
            
            return df
        else:
            print("⚠️ No valid optimization results found")
            return pd.DataFrame()


def main():
    """Command line interface for optimization"""
    parser = argparse.ArgumentParser(description='FSC Meta Hyperparameter Optimization')
    
    parser.add_argument('--model', type=str, choices=['alexnet', 'kan', 'ickan', 'wavkan', 'all'],
                       default='all', help='Model to optimize')
    parser.add_argument('--preset', type=str, 
                       choices=['quick_test', 'development', 'production', 'extensive'],
                       default='development', help='Optimization preset')
    parser.add_argument('--trials', type=int, help='Custom number of trials')
    parser.add_argument('--timeout', type=int, help='Custom timeout in minutes')
    parser.add_argument('--compare', action='store_true', help='Compare existing results')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = FSCMetaOptimizer()
    
    if args.compare:
        optimizer.compare_optimization_results()
        return
    
    if args.model == 'all':
        # Optimize all models
        results = optimizer.optimize_all_models(
            preset=args.preset
        )
    else:
        # Optimize single model
        results = optimizer.optimize_single_model(
            model_name=args.model,
            preset=args.preset,
            custom_trials=args.trials,
            custom_timeout=args.timeout
        )
    
    print("\n🎉 Optimization completed!")
    print("📊 Use --compare flag to see results comparison")


if __name__ == "__main__":
    main()
