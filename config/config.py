"""
Configuration Management for FSC Audio Classification
"""

import yaml
import torch
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Union

@dataclass
class DataConfig:
    """Data loading and preprocessing configuration"""
    data_dir: str = "./data/fsc22"
    pickle_dir: str = "./data/fsc22/Pickle_Files/aug_ts_ps_mel_features_5_20"
    batch_size: int = 16
    num_workers: int = 2
    train_split: float = 0.8
    validation_split: float = 0.2
    stratify: bool = True
    random_state: int = 42

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    architecture: str = "wavkan"  # Architecture type: kan, wavkan, ickan
    model_type: str = "high_performance"  # Model variant: high_performance, basic
    num_classes: int = 26
    input_shape: tuple = (128, 196, 3)
    
    # Model-specific parameters
    hidden_channels: List[int] = None
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    use_residual: bool = True
    activation: str = "SiLU"
    
    def __post_init__(self):
        if self.hidden_channels is None:
            self.hidden_channels = [32, 64, 128, 256]

@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "onecycle"
    
    # Scheduler parameters
    max_lr: float = 1e-2
    pct_start: float = 0.1
    
    # Loss function
    loss_function: str = "CrossEntropyLoss"
    label_smoothing: float = 0.1
    
    # Training settings
    gradient_clipping: float = 1.0
    early_stopping_patience: int = 15
    device: str = "auto"  # auto, cuda, cpu
    
    # Cross-validation
    cross_validation: bool = False
    cv_folds: int = 5
    random_seed: int = 42
    
    # Logging
    print_interval: int = 1
    log_interval: int = 1
    save_best_model: bool = True
    model_save_path: str = "./checkpoints"

@dataclass
class FeatureConfig:
    """Feature extraction configuration"""
    extractor_type: str = "advanced"  # basic, advanced, custom
    feature_dim: int = 256
    use_audio_features: bool = True
    
    # Audio feature parameters
    mel_features: bool = True
    mfcc_features: bool = True
    spectral_features: bool = True
    attention_mechanism: bool = True

@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    features: FeatureConfig
    
    # Experiment settings
    experiment_name: str = "FSC_Audio_Classification"
    seed: int = 42
    verbose: bool = True
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            features=FeatureConfig(**config_dict.get('features', {})),
            **{k: v for k, v in config_dict.items() 
               if k not in ['data', 'model', 'training', 'features']}
        )
    
    def to_yaml(self, config_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'features': self.features.__dict__,
            'experiment_name': self.experiment_name,
            'seed': self.seed,
            'verbose': self.verbose
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_device(self) -> torch.device:
        """Get the appropriate device based on configuration"""
        if self.training.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(self.training.device)

# Configuration loading and saving functions
def load_config(config_path: str) -> Config:
    """Load configuration from YAML file"""
    return Config.from_yaml(config_path)

def save_config(config: Config, config_path: str) -> None:
    """Save configuration to YAML file"""
    config.to_yaml(config_path)

# Default configuration
def get_default_config() -> Config:
    """Get default configuration"""
    return Config(
        data=DataConfig(),
        model=ModelConfig(),
        training=TrainingConfig(),
        features=FeatureConfig()
    )
