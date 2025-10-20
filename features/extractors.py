"""
Feature Extraction Module for Audio Classification
currently not used in the main pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from config.config import FeatureConfig

class BasicAudioFeatureExtractor(nn.Module):
    
    def __init__(self, input_shape: Tuple[int, int, int], feature_dim: int = 128):
        super().__init__()
        h, w, c = input_shape
        
        self.features = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            dummy_out = self.features(dummy)
            flattened_size = dummy_out.numel()
        
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        if x.dim() == 4 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
        
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class AdvancedAudioFeatureExtractor(nn.Module):
    
    def __init__(self, input_shape: Tuple[int, int, int], config: FeatureConfig):
        super().__init__()
        h, w, c = input_shape
        self.config = config
        
        # Multi-scale feature extraction
        self.mel_extractor = self._create_pathway(c, 64, "MEL") if config.mel_features else None
        self.mfcc_extractor = self._create_pathway(c, 64, "MFCC") if config.mfcc_features else None
        self.spectral_extractor = self._create_pathway(c, 64, "Spectral") if config.spectral_features else None
        
        # Calculate number of active pathways
        num_pathways = sum([config.mel_features, config.mfcc_features, config.spectral_features])
        combined_features = num_pathways * 64
        
        # Attention mechanism
        if config.attention_mechanism:
            self.attention = nn.MultiheadAttention(combined_features, 8, batch_first=True)
        
        # Final feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(combined_features, config.feature_dim),
            nn.BatchNorm1d(config.feature_dim),
            nn.SiLU(),
            nn.Dropout(0.2)
        )
        
    def _create_pathway(self, in_channels: int, out_channels: int, pathway_type: str):
        """Create a feature extraction pathway"""
        if pathway_type == "MEL":
            # Optimized for mel-spectrogram features
            return nn.Sequential(
                nn.Conv2d(in_channels, 32, 5, stride=2, padding=2),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(32, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((4, 4))
            )
        elif pathway_type == "MFCC":
            # Optimized for MFCC features
            return nn.Sequential(
                nn.Conv2d(in_channels, 32, (7, 3), stride=2, padding=(3, 1)),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(32, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((4, 4))
            )
        else:  # Spectral
            # Optimized for spectral features
            return nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(32, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((4, 4))
            )
    
    def forward(self, x):
        if x.dim() == 4 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
        
        features = []
        
        # Extract features from each pathway
        if self.mel_extractor is not None:
            mel_feat = self.mel_extractor(x)
            features.append(torch.flatten(mel_feat, 1))
        
        if self.mfcc_extractor is not None:
            mfcc_feat = self.mfcc_extractor(x)
            features.append(torch.flatten(mfcc_feat, 1))
        
        if self.spectral_extractor is not None:
            spectral_feat = self.spectral_extractor(x)
            features.append(torch.flatten(spectral_feat, 1))
        
        # Combine features
        combined = torch.cat(features, dim=1)
        
        # Apply attention if enabled
        if self.config.attention_mechanism:
            # Reshape for attention: (batch, seq_len=1, features)
            combined_reshaped = combined.unsqueeze(1)
            attended, _ = self.attention(combined_reshaped, combined_reshaped, combined_reshaped)
            combined = attended.squeeze(1)
        
        # Final projection
        output = self.feature_projection(combined)
        return output

def create_feature_extractor(input_shape: Tuple[int, int, int], config: FeatureConfig) -> nn.Module:
    
    if config.extractor_type == "basic":
        return BasicAudioFeatureExtractor(input_shape, config.feature_dim)
    elif config.extractor_type == "advanced":
        return AdvancedAudioFeatureExtractor(input_shape, config)
    else:
        raise ValueError(f"Unknown extractor type: {config.extractor_type}")

# For backward compatibility
def create_audio_feature_extractor(input_shape: Tuple[int, int, int], 
                                 feature_dim: int = 256) -> nn.Module:
    """Backward compatibility function"""
    config = FeatureConfig(feature_dim=feature_dim)
    return create_feature_extractor(input_shape, config)
