import yaml
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

class OptimizationConfig:
    
    def __init__(self, config_path: str = None):
        if config_path and Path(config_path).exists():
            self.config = self.load_config(config_path)
        else:
            self.config = self.get_default_config()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            'study': {
                'study_name': 'esc_meta_optimization',
                'storage': 'sqlite:///optimization/optuna_study.db',
                'direction': 'maximize',
                'n_trials': 20,
                'n_jobs': 1,
                'timeout': 3600,
                'load_if_exists': True
            },
            
            'cross_validation': {
                'n_splits': 3,
                'random_state': 42,
                'shuffle': True
            },
            
            'training': {
                'max_epochs': 30,
                'patience': 8,
                'batch_size_options': [32, 64],
                'device': 'auto'
            },
            
            'hyperparameters': {
                'common': {
                    'learning_rate': {
                        'type': 'loguniform',
                        'low': 1e-5,
                        'high': 1e-2
                    },
                    'batch_size': {
                        'type': 'categorical',
                        'choices': [32, 64]
                    },
                    'optimizer': {
                        'type': 'categorical',
                        'choices': ['adam', 'adamw']
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
                        'choices': [256, 512]
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
            
            'pruning': {
                'enabled': True,
                'pruner': 'median',
                'n_startup_trials': 5,
                'n_warmup_steps': 10,
                'interval_steps': 1
            },
            
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
        return self.config['study']
    
    def get_cv_config(self) -> Dict[str, Any]:
        return self.config['cross_validation']
    
    def get_training_config(self) -> Dict[str, Any]:
        return self.config['training']
    
    def get_hyperparameters_for_model(self, model_name: str) -> Dict[str, Any]:
        common = self.config['hyperparameters']['common']
        model_specific = self.config['hyperparameters'].get(model_name.lower(), {})
        return {**common, **model_specific}
    
    def get_pruning_config(self) -> Dict[str, Any]:
        return self.config['pruning']
    
    def get_logging_config(self) -> Dict[str, Any]:
        return self.config['logging']
    
    def update_for_extensive_tuning(self):
        self.config['study']['n_trials'] = 100
        self.config['study']['timeout'] = 7200
        self.config['cross_validation']['n_splits'] = 5
        self.config['training']['max_epochs'] = 50
        self.config['training']['batch_size_options'] = [16, 32, 64, 128]
        
        self.config['hyperparameters']['common']['optimizer']['choices'] = ['adam', 'adamw', 'sgd']
        self.config['hyperparameters']['kan']['hidden_dim']['choices'] = [256, 512, 1024]
        self.config['hyperparameters']['ickan']['hidden_dim']['choices'] = [256, 512, 1024]
        self.config['hyperparameters']['wavkan']['hidden_dim']['choices'] = [256, 512, 1024]
    
    def update_for_quick_testing(self):
        self.config['study']['n_trials'] = 5
        self.config['study']['timeout'] = 300
        self.config['cross_validation']['n_splits'] = 2
        self.config['training']['max_epochs'] = 10
        self.config['training']['patience'] = 3
    
    def save_config(self, output_path: str):
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        print(f"Configuration saved to: {output_path}")
