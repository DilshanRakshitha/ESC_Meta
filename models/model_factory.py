from typing import Tuple


class SimpleModelFactory:
    
    def create_model(self, model_name: str, input_shape: Tuple, num_classes: int):
        
        # ========== CNN ARCHITECTURES ==========
        if model_name == 'alexnet' :
            from models.architectures.AlexNet import AlexNet
            return AlexNet(input_size=3, num_classes=num_classes)
        elif model_name == 'densenet' :
            from models.architectures.DenseNet121 import create_densenet121
            return create_densenet121(num_classes=num_classes, input_channels=3)
        elif model_name == 'efficientnet' :
            from models.architectures.EfficientNetV2B0 import create_efficientnet_v2_b0
            return create_efficientnet_v2_b0(num_classes=num_classes, input_channels=3)
        elif model_name == 'inception' :
            from models.architectures.InceptionV3 import create_inception_v3
            return create_inception_v3(num_classes=num_classes, input_channels=3, pretrained=False)
        elif model_name == 'mobilenet' :
            from models.architectures.MobileNetV3Small import create_mobilenet_v3_small
            return create_mobilenet_v3_small(num_classes=num_classes, input_channels=3)
        elif model_name == 'mobilenetv3large' :
            from models.architectures.MobileNetV3Small import create_mobilenet_v3_large
            return create_mobilenet_v3_large(num_classes=num_classes, input_channels=3)
        elif model_name == 'resnet50' :
            from models.architectures.ResNet50V2 import create_resnet50_v2
            return create_resnet50_v2(num_classes=num_classes, input_channels=3)
        elif model_name == 'resnet18' :
            from models.architectures.ResNet50V2 import create_resnet18
            return create_resnet18(num_classes=num_classes, input_channels=3)
        
        # ========== KAN-INSPIRED ARCHITECTURES (Fast, CNN-based) ==========
        elif model_name == 'kan_inspired' :
            from models.architectures.KAN_inspired_models import create_high_performance_kan_inspired
            return create_high_performance_kan_inspired(input_shape, num_classes)
        elif model_name == 'ickan_inspired' :
            from models.architectures.ICKAN_inspired_models import create_high_performance_ickan_inspired_model
            return create_high_performance_ickan_inspired_model(input_shape, num_classes)
        elif model_name == 'wavkan_inspired' :
            from models.architectures.wavkan_models import create_high_performance_wavkan
            return create_high_performance_wavkan(input_shape, num_classes)
        
        # ========== KAN ARCHITECTURES ==========
        elif model_name == 'kan' :
            from models.architectures.KAN import create_exact_kan
            return create_exact_kan(input_shape, num_classes)
        elif model_name == 'kan_fast' :
            from models.architectures.KAN import create_fast_kan
            return create_fast_kan(input_shape, num_classes, mode='balanced')
        elif model_name == 'kan_memory_safe' :
            from models.architectures.KAN import create_memory_safe_kan
            return create_memory_safe_kan(input_shape, num_classes, max_memory_gb=6)
        
        # ========== ICKAN ARCHITECTURES ==========
        elif model_name == 'ickan':
            from models.architectures.ICKAN import create_ickan
            return create_ickan(input_shape, num_classes, variant="standard")
        elif model_name == 'ickan_light':
            from models.architectures.ICKAN import create_ickan
            return create_ickan(input_shape, num_classes, variant="light")
        elif model_name == 'ickan_deep' :
            from models.architectures.ICKAN import create_ickan
            return create_ickan(input_shape, num_classes, variant="deep")
        
        # ========== RAPID KAN ARCHITECTURES (Fast Learning) ==========
        elif model_name == 'rapid_kan':
            from models.architectures.rapid_kan_models import create_rapid_kan
            return create_rapid_kan(input_shape, num_classes, performance='efficient')
        elif model_name == 'rapid_kan_lite' :
            from models.architectures.rapid_kan_models import create_rapid_kan
            return create_rapid_kan(input_shape, num_classes, performance='lightweight')
        elif model_name == 'rapid_kan_power' :
            from models.architectures.rapid_kan_models import create_rapid_kan
            return create_rapid_kan(input_shape, num_classes, performance='powerful')
        
        else:
            raise ValueError(f"Model {model_name} not supported in cross-validation pipeline")
    