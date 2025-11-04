"""
Optimization Configuration for FSC Meta Hyperparameter Tuning
Defines search spaces and optimization settings for each model
"""

from typing import Dict, Any, List, Tuple

class OptimizationConfig:
    """Configuration class for hyperparameter optimization"""
    
    def __init__(self):
        self.search_spaces = self._define_search_spaces()
        self.optimization_settings = self._define_optimization_settings()
    
    def _define_search_spaces(self) -> Dict[str, Dict[str, Any]]:
        """
        Define hyperparameter search spaces for each model
        """
        return {
            'common': {
                'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True},
                'batch_size': {'type': 'categorical', 'choices': [16, 32, 64, 128]},
                'optimizer': {'type': 'categorical', 'choices': ['adam', 'adamw', 'sgd']},
                'weight_decay': {'type': 'float', 'low': 1e-6, 'high': 1e-2, 'log': True},
                'epochs': {'type': 'int', 'low': 20, 'high': 100},
                'lr_scheduler': {'type': 'categorical', 'choices': ['cosine', 'exponential', 'step', 'none']},
                'gradient_clip': {'type': 'float', 'low': 0.1, 'high': 2.0},
            },
            
            'alexnet': {
                'dropout_rate': {'type': 'float', 'low': 0.1, 'high': 0.7},
                'momentum': {'type': 'float', 'low': 0.8, 'high': 0.99},
                'lr_decay_factor': {'type': 'float', 'low': 0.1, 'high': 0.9},
                'lr_decay_patience': {'type': 'int', 'low': 5, 'high': 15},
                'bn_momentum': {'type': 'float', 'low': 0.01, 'high': 0.3},
            },
            
            'kan': {
                'hidden_dim': {'type': 'categorical', 'choices': [256, 512, 1024, 2048]},
                'kan_layers': {'type': 'int', 'low': 2, 'high': 6},
                'residual_blocks': {'type': 'int', 'low': 2, 'high': 8},
                'activation': {'type': 'categorical', 'choices': ['silu', 'gelu', 'relu', 'mish']},
                'dropout_rate': {'type': 'float', 'low': 0.1, 'high': 0.6},
                'kan_noise_scale': {'type': 'float', 'low': 0.01, 'high': 0.3},
                'spline_order': {'type': 'int', 'low': 2, 'high': 5},
            },
            
            'ickan': {
                'initial_filters': {'type': 'categorical', 'choices': [32, 64, 128]},
                'filter_multiplier': {'type': 'float', 'low': 1.5, 'high': 3.0},
                'attention_heads': {'type': 'int', 'low': 4, 'high': 16},
                'ickan_depth': {'type': 'int', 'low': 3, 'high': 8},
                'dropout_rate': {'type': 'float', 'low': 0.1, 'high': 0.6},
                'attention_dropout': {'type': 'float', 'low': 0.1, 'high': 0.4},
                'connection_type': {'type': 'categorical', 'choices': ['residual', 'dense', 'highway']},
            },
            
            'wavkan': {
                'wavelet_type': {'type': 'categorical', 'choices': ['morlet', 'ricker', 'complex_morlet', 'mexican_hat']},
                'wavelet_scales': {'type': 'int', 'low': 16, 'high': 64},
                'fusion_strategy': {'type': 'categorical', 'choices': ['concat', 'attention', 'weighted', 'gated']},
                'temporal_pooling': {'type': 'categorical', 'choices': ['avg', 'max', 'adaptive', 'attention']},
                'dropout_rate': {'type': 'float', 'low': 0.1, 'high': 0.6},
                'wavelet_layers': {'type': 'int', 'low': 2, 'high': 6},
                'frequency_bins': {'type': 'int', 'low': 64, 'high': 256},
            }
        }
    
    def _define_optimization_settings(self) -> Dict[str, Any]:
        """
        Define optimization settings
        """
        return {
            'default_trials': 50,
            'quick_trials': 20,
            'thorough_trials': 100,
            'timeout_minutes': 60,
            'cv_folds': 3,
            'early_stopping_patience': 5,
            'pruning_enabled': True,
            'pruning_warmup_steps': 10,
            'pruning_startup_trials': 5,
            'study_direction': 'maximize',
            'sampler': 'tpe',  # Tree-structured Parzen Estimator
        }
    
    def get_search_space(self, model_name: str) -> Dict[str, Any]:
        """
        Get combined search space for a model (common + model-specific)
        """
        common_space = self.search_spaces['common']
        model_space = self.search_spaces.get(model_name.lower(), {})
        
        return {**common_space, **model_space}
    
    def get_optimization_settings(self) -> Dict[str, Any]:
        """
        Get optimization settings
        """
        return self.optimization_settings.copy()


class OptimizationPresets:
    """Pre-defined optimization configurations for different scenarios"""
    
    @staticmethod
    def quick_test() -> Dict[str, Any]:
        """Quick test configuration for development"""
        return {
            'n_trials': 10,
            'timeout_minutes': 10,
            'cv_folds': 2,
            'max_epochs': 20,
        }
    
    @staticmethod
    def development() -> Dict[str, Any]:
        """Development configuration"""
        return {
            'n_trials': 25,
            'timeout_minutes': 30,
            'cv_folds': 3,
            'max_epochs': 50,
        }
    
    @staticmethod
    def production() -> Dict[str, Any]:
        """Production configuration for best results"""
        return {
            'n_trials': 100,
            'timeout_minutes': 120,
            'cv_folds': 5,
            'max_epochs': 100,
        }
    
    @staticmethod
    def extensive() -> Dict[str, Any]:
        """Extensive search for research purposes"""
        return {
            'n_trials': 200,
            'timeout_minutes': 300,
            'cv_folds': 5,
            'max_epochs': 150,
        }


# Model-specific optimization strategies
MODEL_STRATEGIES = {
    'alexnet': {
        'priority_params': ['learning_rate', 'dropout_rate', 'batch_size'],
        'optimization_focus': 'convergence_speed',
        'suggested_trials': 30,
    },
    
    'kan': {
        'priority_params': ['learning_rate', 'hidden_dim', 'kan_layers', 'activation'],
        'optimization_focus': 'architecture_search',
        'suggested_trials': 50,
    },
    
    'ickan': {
        'priority_params': ['learning_rate', 'attention_heads', 'ickan_depth', 'filter_multiplier'],
        'optimization_focus': 'attention_mechanism',
        'suggested_trials': 60,
    },
    
    'wavkan': {
        'priority_params': ['learning_rate', 'wavelet_type', 'wavelet_scales', 'fusion_strategy'],
        'optimization_focus': 'wavelet_parameters',
        'suggested_trials': 70,
    }
}

# Advanced search space configurations
ADVANCED_SEARCH_SPACES = {
    'learning_rate_schedules': {
        'cosine_annealing': {
            'T_max': {'type': 'int', 'low': 10, 'high': 100},
            'eta_min': {'type': 'float', 'low': 1e-6, 'high': 1e-4, 'log': True},
        },
        'exponential': {
            'gamma': {'type': 'float', 'low': 0.9, 'high': 0.99},
        },
        'step': {
            'step_size': {'type': 'int', 'low': 10, 'high': 50},
            'gamma': {'type': 'float', 'low': 0.1, 'high': 0.7},
        }
    },
    
    'data_augmentation': {
        'mixup_alpha': {'type': 'float', 'low': 0.1, 'high': 1.0},
        'cutmix_alpha': {'type': 'float', 'low': 0.1, 'high': 1.0},
        'label_smoothing': {'type': 'float', 'low': 0.0, 'high': 0.2},
    },
    
    'regularization': {
        'dropout_schedule': {'type': 'categorical', 'choices': ['constant', 'linear_decay', 'cosine_decay']},
        'droppath_rate': {'type': 'float', 'low': 0.0, 'high': 0.3},
        'spectral_norm': {'type': 'categorical', 'choices': [True, False]},
    }
}
