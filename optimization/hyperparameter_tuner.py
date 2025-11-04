"""
Modular Hyperparameter Tuning System using Optuna
Integrates with existing FSC Meta pipeline for all models (AlexNet, KAN, ICKAN, WavKAN)
"""

import optuna
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class OptunaTuner:
    """
    Modular hyperparameter tuning system for FSC Meta models
    Uses Optuna for sophisticated optimization with pruning and visualization
    """
    
    def __init__(self, 
                 model_factory,
                 data_loader=None,
                 results_dir: str = "optimization/results",
                 study_name: str = None,
                 storage: str = None):
        """
        Initialize the hyperparameter tuner
        
        Args:
            model_factory: Factory to create models
            data_loader: Data loader instance
            results_dir: Directory to save results
            study_name: Name for the optimization study
            storage: Optuna storage (None for in-memory)
        """
        self.model_factory = model_factory
        self.data_loader = data_loader
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.study_name = study_name or "fsc_meta_optimization"
        self.storage = storage
        
        # Performance tracking
        self.best_results = {}
        self.trial_history = []
        
        print(f"🔍 OptunaTuner initialized")
        print(f"   Results directory: {self.results_dir}")
        print(f"   Study name: {self.study_name}")
    
    def suggest_hyperparameters(self, trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a given model
        Modular design allows easy extension for new models
        """
        # Common hyperparameters for all models
        base_params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd']),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.7),
            'epochs': trial.suggest_int('epochs', 20, 100),
        }
        
        # Model-specific hyperparameters
        if model_name.lower() == 'alexnet':
            model_params = {
                'momentum': trial.suggest_float('momentum', 0.8, 0.99),
                'lr_decay_factor': trial.suggest_float('lr_decay_factor', 0.1, 0.9),
                'lr_decay_patience': trial.suggest_int('lr_decay_patience', 5, 15),
            }
        
        elif model_name.lower() == 'kan':
            model_params = {
                'hidden_dim': trial.suggest_categorical('hidden_dim', [256, 512, 1024, 2048]),
                'kan_layers': trial.suggest_int('kan_layers', 2, 6),
                'residual_blocks': trial.suggest_int('residual_blocks', 2, 8),
                'activation': trial.suggest_categorical('activation', ['silu', 'gelu', 'relu']),
            }
        
        elif model_name.lower() == 'ickan':
            model_params = {
                'initial_filters': trial.suggest_categorical('initial_filters', [32, 64, 128]),
                'filter_multiplier': trial.suggest_float('filter_multiplier', 1.5, 3.0),
                'attention_heads': trial.suggest_int('attention_heads', 4, 16),
                'ickan_depth': trial.suggest_int('ickan_depth', 3, 8),
            }
        
        elif model_name.lower() == 'wavkan':
            model_params = {
                'wavelet_type': trial.suggest_categorical('wavelet_type', ['morlet', 'ricker', 'complex_morlet']),
                'wavelet_scales': trial.suggest_int('wavelet_scales', 16, 64),
                'fusion_strategy': trial.suggest_categorical('fusion_strategy', ['concat', 'attention', 'weighted']),
                'temporal_pooling': trial.suggest_categorical('temporal_pooling', ['avg', 'max', 'adaptive']),
            }
        
        else:
            model_params = {}
        
        # Combine all parameters
        all_params = {**base_params, **model_params}
        
        return all_params
    
    def create_optimized_model(self, model_name: str, params: Dict[str, Any], 
                              input_shape: Tuple, num_classes: int):
        """
        Create model with optimized hyperparameters
        """
        try:
            # Extract model-specific parameters for model creation
            if model_name.lower() == 'kan':
                model_config = {
                    'hidden_dim': params.get('hidden_dim', 512),
                    'dropout_rate': params.get('dropout_rate', 0.1),
                }
                # You would modify your KAN creation to accept these parameters
                model = self.model_factory.create_model(model_name, input_shape, num_classes)
                
            elif model_name.lower() in ['ickan', 'wavkan']:
                # Similar configuration for other models
                model = self.model_factory.create_model(model_name, input_shape, num_classes)
            
            else:  # AlexNet and others
                model = self.model_factory.create_model(model_name, input_shape, num_classes)
            
            return model
            
        except Exception as e:
            print(f"⚠️ Model creation failed: {e}")
            return None
    
    def create_optimized_optimizer(self, model, params: Dict[str, Any]):
        """
        Create optimizer with suggested hyperparameters
        """
        optimizer_type = params['optimizer']
        lr = params['learning_rate']
        weight_decay = params['weight_decay']
        
        if optimizer_type == 'adam':
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            momentum = params.get('momentum', 0.9)
            return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def objective_function(self, trial: optuna.Trial, model_name: str, 
                          features: np.ndarray, labels: np.ndarray) -> float:
        """
        Objective function for Optuna optimization
        Returns validation accuracy to maximize
        """
        # Suggest hyperparameters
        params = self.suggest_hyperparameters(trial, model_name)
        
        # Setup
        device = torch.device('cpu')  # Force CPU to avoid memory issues
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        num_classes = len(np.unique(labels_encoded))
        
        # Cross-validation setup
        cv_scores = []
        n_folds = 3  # Reduced for faster optimization
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels_encoded)):
            try:
                # Split data
                X_train = torch.FloatTensor(features[train_idx])
                X_val = torch.FloatTensor(features[val_idx])
                y_train = torch.LongTensor(labels_encoded[train_idx])
                y_val = torch.LongTensor(labels_encoded[val_idx])
                
                # Create model
                model = self.create_optimized_model(model_name, params, features.shape[1:], num_classes)
                if model is None:
                    return 0.0
                
                model = model.to(device)
                
                # Create optimizer
                optimizer = self.create_optimized_optimizer(model, params)
                criterion = nn.CrossEntropyLoss()
                
                # Training with early stopping
                best_val_acc = 0.0
                patience = 5
                patience_counter = 0
                epochs = min(params['epochs'], 30)  # Cap epochs for optimization
                
                for epoch in range(epochs):
                    # Training
                    model.train()
                    train_loss = 0.0
                    train_correct = 0
                    
                    # Mini-batch training
                    batch_size = params['batch_size']
                    n_batches = len(X_train) // batch_size
                    
                    for i in range(0, len(X_train), batch_size):
                        batch_X = X_train[i:i+batch_size].to(device)
                        batch_y = y_train[i:i+batch_size].to(device)
                        
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        train_correct += (predicted == batch_y).sum().item()
                    
                    # Validation
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val.to(device))
                        _, val_predicted = torch.max(val_outputs.data, 1)
                        val_acc = (val_predicted == y_val.to(device)).float().mean().item()
                    
                    # Early stopping
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            break
                    
                    # Optuna pruning
                    trial.report(val_acc, epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                
                cv_scores.append(best_val_acc)
                
            except optuna.TrialPruned:
                raise
            except Exception as e:
                print(f"⚠️ Fold {fold} failed: {e}")
                cv_scores.append(0.0)
        
        # Return mean CV score
        mean_score = np.mean(cv_scores) if cv_scores else 0.0
        return mean_score
    
    def optimize_model(self, model_name: str, features: np.ndarray, labels: np.ndarray,
                      n_trials: int = 50, timeout: int = 3600) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a specific model
        
        Args:
            model_name: Name of the model to optimize
            features: Feature data
            labels: Labels
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
        
        Returns:
            Dictionary with optimization results
        """
        print(f"🔍 Starting hyperparameter optimization for {model_name}")
        print(f"   Trials: {n_trials}, Timeout: {timeout}s")
        
        # Create study
        study_name_full = f"{self.study_name}_{model_name}"
        
        study = optuna.create_study(
            study_name=study_name_full,
            direction='maximize',
            storage=self.storage,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Optimize
        def objective(trial):
            return self.objective_function(trial, model_name, features, labels)
        
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Save results
        results = {
            'model_name': model_name,
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'study_name': study_name_full
        }
        
        # Save to file
        results_file = self.results_dir / f"{model_name}_optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save study
        study_file = self.results_dir / f"{model_name}_study.pkl"
        with open(study_file, 'wb') as f:
            pickle.dump(study, f)
        
        print(f"✅ Optimization completed for {model_name}")
        print(f"   Best accuracy: {results['best_value']:.4f}")
        print(f"   Best params saved to: {results_file}")
        
        return results
    
    def optimize_all_models(self, features: np.ndarray, labels: np.ndarray,
                           models: List[str] = None, n_trials: int = 50) -> Dict[str, Dict]:
        """
        Optimize hyperparameters for all models
        """
        if models is None:
            models = ['alexnet', 'kan', 'ickan', 'wavkan']
        
        all_results = {}
        
        for model_name in models:
            print(f"\n{'='*60}")
            print(f"🎯 OPTIMIZING {model_name.upper()}")
            print(f"{'='*60}")
            
            try:
                results = self.optimize_model(model_name, features, labels, n_trials)
                all_results[model_name] = results
            except Exception as e:
                print(f"❌ Optimization failed for {model_name}: {e}")
                all_results[model_name] = {'error': str(e)}
        
        # Save combined results
        combined_file = self.results_dir / "all_models_optimization_results.json"
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Generate summary
        self.generate_optimization_summary(all_results)
        
        return all_results
    
    def generate_optimization_summary(self, results: Dict[str, Dict]):
        """
        Generate optimization summary report
        """
        summary_file = self.results_dir / "optimization_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("FSC META - HYPERPARAMETER OPTIMIZATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            for model_name, result in results.items():
                if 'error' in result:
                    f.write(f"{model_name.upper()}: FAILED - {result['error']}\n\n")
                    continue
                
                f.write(f"{model_name.upper()}:\n")
                f.write(f"  Best Accuracy: {result['best_value']:.4f}\n")
                f.write(f"  Trials: {result['n_trials']}\n")
                f.write(f"  Best Parameters:\n")
                
                for param, value in result['best_params'].items():
                    f.write(f"    {param}: {value}\n")
                f.write("\n")
            
            # Best model
            valid_results = {k: v for k, v in results.items() if 'error' not in v}
            if valid_results:
                best_model = max(valid_results.items(), key=lambda x: x[1]['best_value'])
                f.write(f"BEST MODEL: {best_model[0].upper()}\n")
                f.write(f"BEST ACCURACY: {best_model[1]['best_value']:.4f}\n")
        
        print(f"📊 Optimization summary saved to: {summary_file}")
    
    def load_best_params(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Load best parameters for a model
        """
        results_file = self.results_dir / f"{model_name}_optimization_results.json"
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            return results.get('best_params')
        
        return None
    
    def create_visualization(self, model_name: str):
        """
        Create optimization visualization plots
        """
        try:
            import matplotlib.pyplot as plt
            
            study_file = self.results_dir / f"{model_name}_study.pkl"
            if not study_file.exists():
                print(f"⚠️ Study file not found: {study_file}")
                return
            
            with open(study_file, 'rb') as f:
                study = pickle.load(f)
            
            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Hyperparameter Optimization - {model_name.upper()}')
            
            # Optimization history
            optuna.visualization.matplotlib.plot_optimization_history(study, ax=axes[0, 0])
            axes[0, 0].set_title('Optimization History')
            
            # Parameter importance
            optuna.visualization.matplotlib.plot_param_importances(study, ax=axes[0, 1])
            axes[0, 1].set_title('Parameter Importance')
            
            # Parallel coordinate plot
            optuna.visualization.matplotlib.plot_parallel_coordinate(study, ax=axes[1, 0])
            axes[1, 0].set_title('Parallel Coordinate')
            
            # Slice plot for learning rate
            optuna.visualization.matplotlib.plot_slice(study, params=['learning_rate'], ax=axes[1, 1])
            axes[1, 1].set_title('Learning Rate Impact')
            
            plt.tight_layout()
            plot_file = self.results_dir / f"{model_name}_optimization_plots.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Visualization saved to: {plot_file}")
            
        except ImportError:
            print("⚠️ Matplotlib not available for visualization")
        except Exception as e:
            print(f"⚠️ Visualization failed: {e}")
