"""
Hyperparameter Optimizer using Optuna
Main optimization interface
"""

import optuna
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Callable, Tuple, Optional
import logging
import pickle
import pandas as pd

from .optimization_config import OptimizationConfig
from .objective_function import ObjectiveFunction

class HyperparameterOptimizer:
    """Main hyperparameter optimization class"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize optimizer
        
        Args:
            config_path: Path to optimization configuration file
        """
        self.config = OptimizationConfig(config_path)
        self.study = None
        self.best_params = None
        
        # Setup logging
        self._setup_logging()
        
        # Create output directories
        self._create_output_dirs()
        
        print("🔧 Hyperparameter Optimizer initialized")
        print(f"   Study: {self.config.get_study_config()['study_name']}")
        print(f"   Max trials: {self.config.get_study_config()['n_trials']}")
        print(f"   CV folds: {self.config.get_cv_config()['n_splits']}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get_logging_config()
        
        # Create logs directory if it doesn't exist
        log_dir = Path(log_config['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_config['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'optimization.log'),
                logging.StreamHandler()
            ]
        )
        
        # Suppress optuna logs for cleaner output
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def _create_output_dirs(self):
        """Create output directories"""
        log_config = self.config.get_logging_config()
        
        Path(log_config['log_dir']).mkdir(parents=True, exist_ok=True)
        Path(log_config['plot_dir']).mkdir(parents=True, exist_ok=True)
        Path('optimization').mkdir(parents=True, exist_ok=True)
    
    def optimize(self, 
                 model_factory: Callable,
                 data: Tuple[np.ndarray, np.ndarray],
                 model_name: str = 'model') -> Dict[str, Any]:
        """
        Optimize hyperparameters for a given model
        
        Args:
            model_factory: Function that creates model given hyperparameters
            data: Tuple of (features, labels)
            model_name: Name of the model for logging
            
        Returns:
            Dictionary with optimization results
        """
        print(f"\n🚀 Starting hyperparameter optimization for {model_name}")
        print("=" * 60)
        
        # Create study
        self._create_study(model_name)
        
        # Create objective function
        objective = ObjectiveFunction(model_factory, data, self.config)
        
        # Configure pruner if enabled
        study_config = self.config.get_study_config()
        
        try:
            # Run optimization
            print(f"🔍 Running {study_config['n_trials']} trials...")
            
            self.study.optimize(
                objective,
                n_trials=study_config['n_trials'],
                timeout=study_config.get('timeout'),
                n_jobs=study_config.get('n_jobs', 1),
                show_progress_bar=True
            )
            
            # Get results
            results = self._process_results(model_name)
            
            # Save results
            self._save_results(results, model_name)
            
            # Generate plots
            if self.config.get_logging_config()['save_plots']:
                self._generate_plots(model_name)
            
            return results
            
        except Exception as e:
            logging.error(f"Optimization failed: {e}")
            return {'error': str(e)}
    
    def _create_study(self, model_name: str):
        """Create or load Optuna study"""
        study_config = self.config.get_study_config()
        pruning_config = self.config.get_pruning_config()
        
        # Create pruner if enabled
        pruner = None
        if pruning_config['enabled']:
            if pruning_config['pruner'] == 'median':
                pruner = optuna.pruners.MedianPruner(
                    n_startup_trials=pruning_config['n_startup_trials'],
                    n_warmup_steps=pruning_config['n_warmup_steps'],
                    interval_steps=pruning_config['interval_steps']
                )
            elif pruning_config['pruner'] == 'successive_halving':
                pruner = optuna.pruners.SuccessiveHalvingPruner()
            elif pruning_config['pruner'] == 'hyperband':
                pruner = optuna.pruners.HyperbandPruner()
        
        # Create study
        study_name = f"{study_config['study_name']}_{model_name}"
        
        self.study = optuna.create_study(
            study_name=study_name,
            storage=study_config.get('storage'),
            direction=study_config['direction'],
            pruner=pruner,
            load_if_exists=study_config.get('load_if_exists', True)
        )
        
        print(f"📊 Study created: {study_name}")
        if pruner:
            print(f"✂️ Pruner: {pruning_config['pruner']}")
    
    def _process_results(self, model_name: str) -> Dict[str, Any]:
        """Process optimization results"""
        if not self.study.best_trial:
            return {'error': 'No successful trials'}
        
        best_trial = self.study.best_trial
        self.best_params = best_trial.params
        
        # Get trial statistics
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]
        
        results = {
            'model_name': model_name,
            'best_params': self.best_params,
            'best_score': best_trial.value,
            'best_trial_number': best_trial.number,
            'n_trials': len(self.study.trials),
            'completed_trials': len(completed_trials),
            'pruned_trials': len(pruned_trials),
            'cv_scores': best_trial.user_attrs.get('cv_scores', []),
            'cv_std': best_trial.user_attrs.get('std_accuracy', 0),
            'study_name': self.study.study_name
        }
        
        # Print summary
        print(f"\n✅ Optimization completed!")
        print(f"   Best score: {results['best_score']:.4f} ± {results['cv_std']:.4f}")
        print(f"   Best trial: #{results['best_trial_number']}")
        print(f"   Completed trials: {results['completed_trials']}")
        print(f"   Pruned trials: {results['pruned_trials']}")
        print(f"\n🏆 Best hyperparameters:")
        for param, value in self.best_params.items():
            print(f"   {param}: {value}")
        
        return results
    
    def _save_results(self, results: Dict[str, Any], model_name: str):
        """Save optimization results"""
        log_config = self.config.get_logging_config()
        
        if log_config['save_best_params']:
            # Save best parameters
            params_file = Path(log_config['log_dir']) / f'{model_name}_best_params.yaml'
            import yaml
            with open(params_file, 'w') as f:
                yaml.dump(results['best_params'], f, default_flow_style=False)
            print(f"💾 Best parameters saved: {params_file}")
        
        if log_config['save_study']:
            # Save full results
            results_file = Path(log_config['log_dir']) / f'{model_name}_optimization_results.pkl'
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)
            print(f"💾 Results saved: {results_file}")
            
            # Save trials dataframe
            df = self.study.trials_dataframe()
            csv_file = Path(log_config['log_dir']) / f'{model_name}_trials.csv'
            df.to_csv(csv_file, index=False)
            print(f"📊 Trials saved: {csv_file}")
    
    def _generate_plots(self, model_name: str):
        """Generate optimization plots"""
        plot_dir = Path(self.config.get_logging_config()['plot_dir'])
        
        try:
            # Optimization history
            fig = optuna.visualization.matplotlib.plot_optimization_history(self.study)
            plt.title(f'Optimization History - {model_name}')
            plt.savefig(plot_dir / f'{model_name}_optimization_history.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Parameter importances
            if len(self.study.trials) > 10:
                fig = optuna.visualization.matplotlib.plot_param_importances(self.study)
                plt.title(f'Parameter Importances - {model_name}')
                plt.savefig(plot_dir / f'{model_name}_param_importances.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # Parallel coordinate plot
            if len(self.best_params) > 1:
                fig = optuna.visualization.matplotlib.plot_parallel_coordinate(self.study)
                plt.title(f'Parallel Coordinate Plot - {model_name}')
                plt.savefig(plot_dir / f'{model_name}_parallel_coordinate.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"📈 Plots saved in: {plot_dir}")
            
        except Exception as e:
            logging.warning(f"Could not generate plots: {e}")
    
    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get best parameters from last optimization"""
        return self.best_params
    
    def load_study(self, study_name: str):
        """Load existing study"""
        study_config = self.config.get_study_config()
        
        self.study = optuna.load_study(
            study_name=study_name,
            storage=study_config.get('storage')
        )
        
        if self.study.best_trial:
            self.best_params = self.study.best_trial.params
        
        print(f"📂 Study loaded: {study_name}")
        print(f"   Trials: {len(self.study.trials)}")
        if self.best_params:
            print(f"   Best score: {self.study.best_value:.4f}")
    
    def quick_test_mode(self):
        """Switch to quick testing mode"""
        print("⚡ Switching to quick test mode...")
        self.config.update_for_quick_testing()
    
    def extensive_tuning_mode(self):
        """Switch to extensive tuning mode"""
        print("🔥 Switching to extensive tuning mode...")
        self.config.update_for_extensive_tuning()
