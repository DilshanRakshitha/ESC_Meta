from typing import Tuple

from models.architectures.AlexNet import AlexNet
from models.architectures.kan_models import create_high_performance_kan
from models.architectures.ickan_models import create_high_performance_ickan
from models.architectures.wavkan_models import create_high_performance_wavkan
from models.architectures.exact_kan_models import create_exact_kan
from models.architectures.exact_kan_models import create_fast_exact_kan, create_memory_safe_kan
from models.architectures.exact_ickan_models import create_exact_ickan, create_ickan_model
from models.architectures.rapid_kan_models import create_rapid_kan
from models.architectures.DenseNet121 import create_densenet121
from models.architectures.EfficientNetV2B0 import create_efficientnet_v2_b0
from models.architectures.InceptionV3 import create_inception_v3
from models.architectures.ResNet50V2 import create_resnet50_v2, create_resnet18
from models.architectures.MobileNetV3Small import create_mobilenet_v3_small, create_mobilenet_v3_large


class SimpleModelFactory:
    
    def create_model(self, model_name: str, input_shape: Tuple, num_classes: int):
        
        if model_name.lower() == 'alexnet':
            input_channels = input_shape[0] if len(input_shape) == 3 else 3
            return AlexNet(input_size=input_channels, num_classes=num_classes)

        elif model_name.lower() == 'densenet':
            input_channels = input_shape[0] if len(input_shape) == 3 else 3
            return create_densenet121(num_classes=num_classes, input_channels=input_channels)

        elif model_name.lower() == 'efficientnet':
            input_channels = input_shape[0] if len(input_shape) == 3 else 3
            return create_efficientnet_v2_b0(num_classes=num_classes, input_channels=input_channels)

        elif model_name.lower() == 'inception':
            input_channels = input_shape[0] if len(input_shape) == 3 else 3
            return create_inception_v3(num_classes=num_classes, input_channels=input_channels)

        elif model_name.lower() == 'resnet50':
            input_channels = input_shape[0] if len(input_shape) == 3 else 3
            return create_resnet50_v2(num_classes=num_classes, input_channels=input_channels)

        elif model_name.lower() == 'resnet18':
            input_channels = input_shape[0] if len(input_shape) == 3 else 3
            return create_resnet18(num_classes=num_classes, input_channels=input_channels)

        elif model_name.lower() == 'mobilenetv3small':
            input_channels = input_shape[0] if len(input_shape) == 3 else 3
            return create_mobilenet_v3_small(num_classes=num_classes, input_channels=input_channels)

        elif model_name.lower() == 'mobilenetv3large':
            input_channels = input_shape[0] if len(input_shape) == 3 else 3
            return create_mobilenet_v3_large(num_classes=num_classes, input_channels=input_channels)
            
        elif model_name.lower() == 'kan':
            if len(input_shape) == 3:  # (C, H, W) -> (H, W, C)
                kan_input_shape = (input_shape[1], input_shape[2], input_shape[0])
            else:
                kan_input_shape = input_shape
            return create_high_performance_kan(kan_input_shape, num_classes)
            
        elif model_name.lower() == 'ickan':
            if len(input_shape) == 3:  # (C, H, W) -> (H, W, C)
                ickan_input_shape = (input_shape[1], input_shape[2], input_shape[0])
            else:
                ickan_input_shape = input_shape
            return create_high_performance_ickan(ickan_input_shape, num_classes)
            
        elif model_name.lower() == 'wavkan':
            if len(input_shape) == 3:  # (C, H, W) -> (H, W, C)
                wavkan_input_shape = (input_shape[1], input_shape[2], input_shape[0])
            else:
                wavkan_input_shape = input_shape
            return create_high_performance_wavkan(wavkan_input_shape, num_classes)
            
        elif model_name.lower() == 'exact_kan': # will need a lot of computation
            if len(input_shape) == 3:  # (C, H, W) -> (H, W, C)
                exact_kan_input_shape = (input_shape[1], input_shape[2], input_shape[0])
            else:
                exact_kan_input_shape = input_shape
            return create_exact_kan(exact_kan_input_shape, num_classes)

        elif model_name.lower() == 'fast_exact_kan': # 87
            # Balanced accuracy and regularization
            if len(input_shape) == 3:
                fast_input_shape = (input_shape[1], input_shape[2], input_shape[0])
            else:
                fast_input_shape = input_shape
            return create_fast_exact_kan(fast_input_shape, num_classes, mode='balanced')
            
        elif model_name.lower() == 'fast_exact_kan_heavy': # 85
            # High regularization for overfitting
            if len(input_shape) == 3:
                fast_input_shape = (input_shape[1], input_shape[2], input_shape[0])
            else:
                fast_input_shape = input_shape
            return create_fast_exact_kan(fast_input_shape, num_classes, mode='regularized')
            
        elif model_name.lower() == 'fast_exact_kan_max': # 83 and increasing
            # Memory-efficient high accuracy (fixed the crash issue)
            if len(input_shape) == 3:
                fast_input_shape = (input_shape[1], input_shape[2], input_shape[0])
            else:
                fast_input_shape = input_shape
            return create_fast_exact_kan(fast_input_shape, num_classes, mode='high_accuracy')
            
        elif model_name.lower() == 'memory_safe_kan': #89 and increasing
            # Ultra-safe version that won't crash your system
            if len(input_shape) == 3:
                safe_input_shape = (input_shape[1], input_shape[2], input_shape[0])
            else:
                safe_input_shape = input_shape
            return create_memory_safe_kan(safe_input_shape, num_classes, max_memory_gb=6)
            
        elif model_name.lower() == 'exact_ickan': # 75
            if len(input_shape) == 3:  # (C, H, W) -> (H, W, C)
                exact_ickan_input_shape = (input_shape[1], input_shape[2], input_shape[0])
            else:
                exact_ickan_input_shape = input_shape
            return create_exact_ickan(exact_ickan_input_shape, num_classes, variant="standard")
            
        elif model_name.lower() == 'light_ickan': # 67 get overfitted
            if len(input_shape) == 3:  # (C, H, W) -> (H, W, C)
                light_ickan_input_shape = (input_shape[1], input_shape[2], input_shape[0])
            else:
                light_ickan_input_shape = input_shape
            return create_exact_ickan(light_ickan_input_shape, num_classes, variant="light")
            
        elif model_name.lower() == 'deep_ickan': # 89
            if len(input_shape) == 3:  # (C, H, W) -> (H, W, C)
                deep_ickan_input_shape = (input_shape[1], input_shape[2], input_shape[0])
            else:
                deep_ickan_input_shape = input_shape
            return create_exact_ickan(deep_ickan_input_shape, num_classes, variant="deep")
            
        elif model_name.lower() == 'rapid_kan': # 97
            # Fast-learning KAN optimized for quick convergence
            if len(input_shape) == 3:  # (C, H, W) -> (H, W, C)
                rapid_input_shape = (input_shape[1], input_shape[2], input_shape[0])
            else:
                rapid_input_shape = input_shape
            return create_rapid_kan(rapid_input_shape, num_classes, performance='efficient')
            
        elif model_name.lower() == 'rapid_kan_lite': # 98
            # Lightweight fast-learning KAN
            if len(input_shape) == 3:  # (C, H, W) -> (H, W, C)
                lite_input_shape = (input_shape[1], input_shape[2], input_shape[0])
            else:
                lite_input_shape = input_shape
            return create_rapid_kan(lite_input_shape, num_classes, performance='lightweight')
            
        elif model_name.lower() == 'rapid_kan_power': # 98
            # Powerful fast-learning KAN
            if len(input_shape) == 3:  # (C, H, W) -> (H, W, C)
                power_input_shape = (input_shape[1], input_shape[2], input_shape[0])
            else:
                power_input_shape = input_shape
            return create_rapid_kan(power_input_shape, num_classes, performance='powerful')
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
