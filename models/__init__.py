"""
ESC_Meta Models Package - Modular KAN Architecture System
Provides access to KAN, WavKAN, ICKAN model architectures and advanced training utilities
"""

# Import our new modular architectures
try:
    from .architectures import (
        create_kan_inspired_model, HighPerformanceKANinspired, BasicKANinpired,
        create_wavkan_model, HighPerformanceWavKAN, BasicWavKAN,
        create_ickan_inspired_models, HighPerformanceICKAN, BasicICKAN
    )
except ImportError as e:
    print(f"Warning: Could not import modular architectures: {e}")
    create_kan_inspired_model = HighPerformanceKANinspired = BasicKANinpired = None
    create_wavkan_model = HighPerformanceWavKAN = BasicWavKAN = None
    create_ickan_inspired_models = HighPerformanceICKAN = BasicICKAN = None

# Import new training utilities
try:
    from .training import (
        AdvancedTrainer, KFoldTrainer, create_trainer, FocalLoss,
        ModelTrainer, AudioDataset
    )
except ImportError as e:
    print(f"Warning: Could not import training utilities: {e}")
    AdvancedTrainer = KFoldTrainer = create_trainer = FocalLoss = None
    ModelTrainer = AudioDataset = None

# Legacy imports (keep for backward compatibility but make optional)
try:
    from .architectures.AlexNet import AlexNet
except ImportError:
    AlexNet = None

__all__ = [
    # New modular KAN architectures
    'create_kan_inspired_model', 'HighPerformanceKANinspired', 'BasicKANinpired',
    'create_wavkan_model', 'HighPerformanceWavKAN', 'BasicWavKAN',
    'create_ickan_inspired_models', 'HighPerformanceICKAN', 'BasicICKAN',
    
    # New training utilities
    'AdvancedTrainer', 'KFoldTrainer', 'create_trainer', 'FocalLoss',
    
    # Legacy compatibility
    'ModelTrainer', 'AudioDataset', 'AlexNet'
    'KnowledgeDistillation',
    'ModelCompressionPipeline'
]
