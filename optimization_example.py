"""
FSC Meta Optimization Integration Example
Shows how to integrate hyperparameter optimization with existing models
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.extend([
    str(project_root),
    str(project_root / 'optimization'),
])

# Import existing FSC Meta components
from main import SimpleModelFactory, FSCMetaMain
from optimization import FSCMetaOptimizer, quick_optimize, production_optimize


class OptimizedFSCPipeline:
    """
    Enhanced FSC pipeline with integrated hyperparameter optimization
    """
    
    def __init__(self):
        """Initialize pipeline with optimization capabilities"""
        
        # Initialize existing components
        self.main_pipeline = FSCMetaMain()
        self.model_factory = SimpleModelFactory()
        
        # Initialize optimizer
        self.optimizer = FSCMetaOptimizer(
            model_factory=self.model_factory,
            data_loader=self.main_pipeline.data_loader
        )
        
        print("🔗 Optimized FSC Pipeline initialized")
        print("   ✅ Existing pipeline integrated")
        print("   ✅ Optimization system ready")
    
    def run_optimization_workflow(self, model_name: str = 'alexnet', mode: str = 'quick'):
        """
        Complete optimization workflow
        
        Args:
            model_name: Model to optimize ('alexnet', 'kan', 'ickan', 'wavkan')
            mode: Optimization mode ('quick', 'development', 'production')
        """
        
        print(f"🚀 OPTIMIZATION WORKFLOW - {model_name.upper()}")
        print("="*60)
        
        # Step 1: Load data
        print("📂 Step 1: Loading data...")
        features, labels = self.main_pipeline.load_data('auto')
        print(f"   Data shape: {features.shape}")
        print(f"   Classes: {len(np.unique(labels))}")
        
        # Step 2: Baseline performance (before optimization)
        print("\n🔧 Step 2: Testing baseline performance...")
        baseline_result = self._test_baseline_model(model_name, features, labels)
        
        # Step 3: Run optimization
        print(f"\n🔍 Step 3: Running {mode} optimization...")
        if mode == 'quick':
            optimization_result = self.optimizer.optimize_single_model(
                model_name=model_name,
                preset='quick_test',
                custom_trials=5
            )
        elif mode == 'development':
            optimization_result = self.optimizer.optimize_single_model(
                model_name=model_name,
                preset='development'
            )
        elif mode == 'production':
            optimization_result = self.optimizer.optimize_single_model(
                model_name=model_name,
                preset='production'
            )
        
        # Step 4: Compare results
        print("\n📊 Step 4: Results comparison...")
        self._compare_results(baseline_result, optimization_result)
        
        # Step 5: Load optimized model
        print("\n🎯 Step 5: Loading optimized model...")
        optimized_model, best_params = self.optimizer.load_optimized_model(
            model_name, features.shape[1:], len(np.unique(labels))
        )
        
        return {
            'baseline': baseline_result,
            'optimized': optimization_result,
            'model': optimized_model,
            'best_params': best_params
        }
    
    def _test_baseline_model(self, model_name: str, features: np.ndarray, labels: np.ndarray) -> dict:
        """Test baseline model performance"""
        
        try:
            # Use existing training method from main pipeline
            result = self.main_pipeline.train_model_simple(features, labels, model_name)
            
            baseline_accuracy = result['mean_accuracy']
            print(f"   Baseline accuracy: {baseline_accuracy:.2f}%")
            
            return {
                'accuracy': baseline_accuracy,
                'method': 'baseline',
                'params': 'default'
            }
            
        except Exception as e:
            print(f"   ⚠️ Baseline test failed: {e}")
            return {
                'accuracy': 0.0,
                'method': 'baseline_failed',
                'params': 'default'
            }
    
    def _compare_results(self, baseline: dict, optimized: dict):
        """Compare baseline vs optimized results"""
        
        baseline_acc = baseline.get('accuracy', 0.0)
        optimized_acc = optimized.get('best_value', 0.0) * 100  # Convert to percentage
        
        improvement = optimized_acc - baseline_acc
        improvement_pct = (improvement / baseline_acc * 100) if baseline_acc > 0 else 0
        
        print(f"   📈 PERFORMANCE COMPARISON")
        print(f"   Baseline:  {baseline_acc:.2f}%")
        print(f"   Optimized: {optimized_acc:.2f}%")
        print(f"   Improvement: {improvement:+.2f}% ({improvement_pct:+.1f}%)")
        
        if improvement > 0:
            print("   🎉 Optimization successful!")
        else:
            print("   ⚠️ Optimization did not improve performance")
    
    def run_comprehensive_optimization(self):
        """
        Run comprehensive optimization for all models
        """
        print("🚀 COMPREHENSIVE MODEL OPTIMIZATION")
        print("="*80)
        
        models = ['alexnet', 'kan', 'ickan', 'wavkan']
        results = {}
        
        for model_name in models:
            print(f"\n🎯 Optimizing {model_name.upper()}...")
            try:
                result = self.run_optimization_workflow(model_name, mode='development')
                results[model_name] = result
            except Exception as e:
                print(f"❌ Failed to optimize {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        # Generate summary
        self._generate_summary_report(results)
        
        return results
    
    def _generate_summary_report(self, results: dict):
        """Generate comprehensive summary report"""
        
        print("\n📊 OPTIMIZATION SUMMARY REPORT")
        print("="*80)
        
        successful_results = []
        
        for model_name, result in results.items():
            if 'error' in result:
                print(f"{model_name.upper()}: ❌ Failed - {result['error']}")
            else:
                baseline_acc = result['baseline'].get('accuracy', 0.0)
                optimized_acc = result['optimized'].get('best_value', 0.0) * 100
                improvement = optimized_acc - baseline_acc
                
                print(f"{model_name.upper()}:")
                print(f"  Baseline: {baseline_acc:.2f}%")
                print(f"  Optimized: {optimized_acc:.2f}%")
                print(f"  Improvement: {improvement:+.2f}%")
                
                successful_results.append((model_name, optimized_acc, improvement))
        
        if successful_results:
            # Best model
            best_model = max(successful_results, key=lambda x: x[1])
            print(f"\n🏆 BEST MODEL: {best_model[0].upper()}")
            print(f"   Accuracy: {best_model[1]:.2f}%")
            print(f"   Improvement: {best_model[2]:+.2f}%")


def demo_quick_optimization():
    """Quick demonstration of optimization system"""
    
    print("🎬 FSC Meta Optimization Demo")
    print("="*50)
    
    # Initialize pipeline
    pipeline = OptimizedFSCPipeline()
    
    # Quick optimization test
    result = pipeline.run_optimization_workflow('alexnet', mode='quick')
    
    print("\n✅ Demo completed!")
    return result


def demo_development_optimization():
    """Development-level optimization demo"""
    
    print("🔬 FSC Meta Development Optimization")
    print("="*50)
    
    pipeline = OptimizedFSCPipeline()
    
    # Development optimization
    result = pipeline.run_optimization_workflow('kan', mode='development')
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FSC Meta Optimization Examples')
    parser.add_argument('--demo', choices=['quick', 'development', 'comprehensive'], 
                       default='quick', help='Demo type to run')
    parser.add_argument('--model', choices=['alexnet', 'kan', 'ickan', 'wavkan'], 
                       default='alexnet', help='Model for single optimization')
    
    args = parser.parse_args()
    
    if args.demo == 'quick':
        demo_quick_optimization()
    elif args.demo == 'development':
        demo_development_optimization()
    elif args.demo == 'comprehensive':
        pipeline = OptimizedFSCPipeline()
        pipeline.run_comprehensive_optimization()
