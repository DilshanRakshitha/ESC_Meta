# Core CNN Models
try:
    from .AlexNet import AlexNet
    from .DenseNet121 import DenseNet121, create_densenet121
    from .EfficientNetV2B0 import EfficientNetV2B0, create_efficientnet_v2_b0
    from .InceptionV3 import InceptionV3, create_inception_v3
    from .MobileNetV3Small import MobileNetV3Small, MobileNetV3Large, create_mobilenet_v3_small, create_mobilenet_v3_large
except ImportError as e:
    print(f"Warning: Could not import CNN models: {e}")

try:
    from .ResNet50V2 import ResNet50V2, ResNet18, create_resnet50_v2, create_resnet18
except ImportError as e:
    print(f"Warning: Could not import ResNet models: {e}")
    ResNet50V2 = ResNet18 = create_resnet50_v2 = create_resnet18 = None

# KAN-based Models
try:
    from .kan_models import create_kan_model, create_high_performance_kan
    print("✅ KAN models imported")
except ImportError as e:
    print(f"Warning: Could not import KAN models: {e}")
    create_kan_model = create_high_performance_kan = None

try:
    from .wavkan_models import create_wavkan_model, create_high_performance_wavkan
    print("✅ WavKAN models imported")
except ImportError as e:
    print(f"Warning: Could not import WavKAN models: {e}")
    create_wavkan_model = create_high_performance_wavkan = None

try:
    from .ickan_models import create_ickan_model, create_high_performance_ickan
    print("✅ ICKAN models imported")
except ImportError as e:
    print(f"Warning: Could not import ICKAN models: {e}")
    create_ickan_model = create_high_performance_ickan = None
except ImportError as e:
    print(f"Warning: Could not import ICKAN models: {e}")
    create_ickan_model = HighPerformanceICKAN = BasicICKAN = create_high_performance_ickan = None

# Model factory functions
__all__ = [
    # CNN Models
    'AlexNet',
    'DenseNet121', 'create_densenet121',
    'EfficientNetV2B0', 'create_efficientnet_v2_b0',
    'InceptionV3', 'create_inception_v3',
    'ResNet50V2', 'ResNet18', 'create_resnet50_v2', 'create_resnet18',
    'MobileNetV3Small', 'MobileNetV3Large', 'create_mobilenet_v3_small', 'create_mobilenet_v3_large',
    
    # KAN-based Models
    'HighPerformanceKAN', 'BasicKAN', 'create_kan_model', 'create_high_performance_kan',
    'HighPerformanceWavKAN', 'BasicWavKAN', 'create_wavkan_model', 'create_high_performance_wavkan',
    'HighPerformanceICKAN', 'BasicICKAN', 'create_ickan_model', 'create_high_performance_ickan'
]
