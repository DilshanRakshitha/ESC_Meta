from typing import Tuple

from models.architectures.AlexNet import AlexNet
from models.architectures.kan_models import create_high_performance_kan
from models.architectures.ickan_models import create_high_performance_ickan
from models.architectures.wavkan_models import create_high_performance_wavkan
from models.architectures.DenseNet121 import create_densenet121
from models.architectures.EfficientNetV2B0 import create_efficientnet_v2_b0
from models.architectures.InceptionV3 import create_inception_v3
# from models.architectures.ResNet50V2 import create_resnet50_v2, create_resnet18
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

        # elif model_name.lower() == 'resnet50':
        #     input_channels = input_shape[0] if len(input_shape) == 3 else 3
        #     return create_resnet50_v2(num_classes=num_classes, input_channels=input_channels)

        # elif model_name.lower() == 'resnet18':
        #     input_channels = input_shape[0] if len(input_shape) == 3 else 3
        #     return create_resnet18(num_classes=num_classes, input_channels=input_channels)

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
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
