"""
Optimization Configuration
Controls all hyperparameter tuning settings and ranges
"""

import yaml
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

class OptimizationConfig:
    """Configuration class for hyperparameter optimization"""
    
    def __init__(self, config_path: str = None):
        """Initialize with config file or defaults"""
        if config_path and Path(config_path).exists():
            self.config = self.load_config(config_path)
        else:
            self.config = self.get_default_config()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Default optimization configuration with small/fast settings"""
        return {
            # Study settings
            'study': {
                'study_name': 'esc_meta_optimization',
                'storage': 'sqlite:///optimization/optuna_study.db',
                'direction': 'maximize',  # maximize accuracy
                'n_trials': 20,  # Small number for quick testing
                'n_jobs': 1,  # Parallel jobs
                'timeout': 3600,  # 1 hour timeout
                'load_if_exists': True
            },
            
            # Cross-validation settings
            'cross_validation': {
                'n_splits': 3,  # Reduced from 5 for speed
                'random_state': 42,
                'shuffle': True
            },
            
            # Training settings
            'training': {
                'max_epochs': 30,  # Reduced from 50 for speed
                'patience': 8,  # Early stopping patience
                'batch_size_options': [32, 64],  # Limited options
                'device': 'auto'  # 'cpu', 'cuda', or 'auto'
            },
            
            # Hyperparameter ranges for different models
            'hyperparameters': {
                'common': {
                    'learning_rate': {
                        'type': 'loguniform',
                        'low': 1e-5,
                        'high': 1e-2
                    },
                    'batch_size': {
                        'type': 'categorical',
                        'choices': [32, 64]  # Limited choices for speed
                    },
                    'optimizer': {
                        'type': 'categorical',
                        'choices': ['adam', 'adamw']  # Limited optimizers
                    },
                    'weight_decay': {
                        'type': 'loguniform',
                        'low': 1e-6,
                        'high': 1e-3
                    }
                },
                
                'alexnet': {
                    'dropout_rate': {
                        'type': 'uniform',
                        'low': 0.3,
                        'high': 0.7
                    }
                },
                
                'kan': {
                    'hidden_dim': {
                        'type': 'categorical',
                        'choices': [256, 512]  # Limited choices
                    },
                    'dropout_rate': {
                        'type': 'uniform',
                        'low': 0.1,
                        'high': 0.5
                    }
                },
                
                'ickan': {
                    'hidden_dim': {
                        'type': 'categorical',
                        'choices': [256, 512]
                    },
                    'dropout_rate': {
                        'type': 'uniform',
                        'low': 0.1,
                        'high': 0.5
                    },
                    'icassp_layers': {
                        'type': 'int',
                        'low': 2,
                        'high': 4
                    }
                },
                
                'wavkan': {
                    'hidden_dim': {
                        'type': 'categorical',
                        'choices': [256, 512]
                    },
                    'dropout_rate': {
                        'type': 'uniform',
                        'low': 0.1,
                        'high': 0.5
                    },
                    'wavelet_type': {
                        'type': 'categorical',
                        'choices': ['morlet', 'mexican_hat']
                    }
                }
            },
            
            # Pruning settings
            'pruning': {
                'enabled': True,
                'pruner': 'median',  # 'median', 'successive_halving', or 'hyperband'
                'n_startup_trials': 5,
                'n_warmup_steps': 10,
                'interval_steps': 1
            },
            
            # Logging and output settings
            'logging': {
                'log_level': 'INFO',
                'save_plots': True,
                'plot_dir': 'optimization/plots',
                'log_dir': 'optimization/logs',
                'save_best_params': True,
                'save_study': True
            }
        }
    
    def get_study_config(self) -> Dict[str, Any]:
        """Get study configuration"""
        return self.config['study']
    
    def get_cv_config(self) -> Dict[str, Any]:
        """Get cross-validation configuration"""
        return self.config['cross_validation']
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config['training']
    
    def get_hyperparameters_for_model(self, model_name: str) -> Dict[str, Any]:
        """Get hyperparameter ranges for specific model"""
        common = self.config['hyperparameters']['common']
        model_specific = self.config['hyperparameters'].get(model_name.lower(), {})
        return {**common, **model_specific}
    
    def get_pruning_config(self) -> Dict[str, Any]:
        """Get pruning configuration"""
        return self.config['pruning']
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.config['logging']
    
    def update_for_extensive_tuning(self):
        """Update config for extensive hyperparameter tuning"""
        self.config['study']['n_trials'] = 100
        self.config['study']['timeout'] = 7200  # 2 hours
        self.config['cross_validation']['n_splits'] = 5
        self.config['training']['max_epochs'] = 50
        self.config['training']['batch_size_options'] = [16, 32, 64, 128]
        
        # Expand hyperparameter ranges
        self.config['hyperparameters']['common']['optimizer']['choices'] = ['adam', 'adamw', 'sgd']
        self.config['hyperparameters']['kan']['hidden_dim']['choices'] = [256, 512, 1024]
        self.config['hyperparameters']['ickan']['hidden_dim']['choices'] = [256, 512, 1024]
        self.config['hyperparameters']['wavkan']['hidden_dim']['choices'] = [256, 512, 1024]
    
    def update_for_quick_testing(self):
        """Update config for quick testing"""
        self.config['study']['n_trials'] = 5
        self.config['study']['timeout'] = 300  # 5 minutes
        self.config['cross_validation']['n_splits'] = 2
        self.config['training']['max_epochs'] = 10
        self.config['training']['patience'] = 3
    
    def save_config(self, output_path: str):
        """Save current configuration to file"""
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        print(f"✅ Configuration saved to: {output_path}")
