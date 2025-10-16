"""
Data Preprocessing Utilities for Audio Classification
"""

import random
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

def random_crop(sound, size):
    """Randomly crop sound to specified size"""
    org_size = len(sound)
    start = random.randint(0, org_size - size)
    return sound[start: start + size]

def padding(sound, size):
    """Pad sound to specified size"""
    diff = size - len(sound)
    return np.pad(sound, (diff//2, diff-(diff//2)), 'constant')

class DataPreprocessor:
    """Data preprocessing for FSC22 dataset"""
    
    def __init__(self, data_path: str):
        """Initialize with data path"""
        self.data_path = Path(data_path)
        
    def prepare_fsc22_data(self, cache: bool = True) -> Tuple[List, List]:
        """Load and prepare FSC22 dataset"""
        print(f"Loading FSC22 data from {self.data_path}")
        
        features = []
        labels = []
        
        # Try to load from all available folds
        for fold in range(1, 6):
            fold_file = self.data_path / f"aug_ts_ps_mel_features_5_20_fold{fold}"
            
            if fold_file.exists():
                try:
                    with open(fold_file, 'rb') as f:
                        fold_data = pickle.load(f)
                        
                    # Extract features and labels from fold
                    if isinstance(fold_data, dict):
                        if 'features' in fold_data and 'labels' in fold_data:
                            features.extend(fold_data['features'])
                            labels.extend(fold_data['labels'])
                        else:
                            # Try different key formats
                            for key in fold_data.keys():
                                if 'feature' in key.lower():
                                    features.extend(fold_data[key])
                                elif 'label' in key.lower():
                                    labels.extend(fold_data[key])
                    elif isinstance(fold_data, list):
                        # Assume it's a list of [features, labels] pairs
                        for item in fold_data:
                            if len(item) == 2:
                                features.append(item[0])
                                labels.append(item[1])
                    
                    print(f"Loaded fold {fold}: {len(fold_data)} samples")
                    
                except Exception as e:
                    print(f"Warning: Could not load fold {fold}: {e}")
                    continue
            else:
                print(f"Fold {fold} file not found: {fold_file}")
        
        print(f"Total loaded: {len(features)} samples, {len(labels)} labels")
        
        # Convert to numpy arrays if needed
        if len(features) > 0:
            if not isinstance(features[0], np.ndarray):
                features = [np.array(f) for f in features]
            if not isinstance(labels[0], np.ndarray):
                labels = [np.array(l) if not isinstance(l, (int, float)) else l for l in labels]
        
        return features, labels
    
    def preprocess_features(self, features: List[np.ndarray], 
                          target_shape: Optional[Tuple] = None) -> List[np.ndarray]:
        """Preprocess features to target shape"""
        if target_shape is None:
            return features
            
        processed = []
        for feature in features:
            # Resize or pad to target shape
            if feature.shape != target_shape:
                # Simple resizing logic - can be enhanced
                processed.append(self._resize_feature(feature, target_shape))
            else:
                processed.append(feature)
        
        return processed
    
    def _resize_feature(self, feature: np.ndarray, target_shape: Tuple) -> np.ndarray:
        """Resize feature to target shape"""
        # Simple padding/cropping - can be enhanced with proper resizing
        current_shape = feature.shape
        
        if len(current_shape) == len(target_shape):
            # Same dimensionality
            result = feature.copy()
            
            for dim in range(len(target_shape)):
                if current_shape[dim] < target_shape[dim]:
                    # Pad
                    pad_amount = target_shape[dim] - current_shape[dim]
                    pad_before = pad_amount // 2
                    pad_after = pad_amount - pad_before
                    
                    pad_width = [(0, 0)] * len(current_shape)
                    pad_width[dim] = (pad_before, pad_after)
                    
                    result = np.pad(result, pad_width, mode='constant')
                elif current_shape[dim] > target_shape[dim]:
                    # Crop
                    start = (current_shape[dim] - target_shape[dim]) // 2
                    
                    indices = [slice(None)] * len(current_shape)
                    indices[dim] = slice(start, start + target_shape[dim])
                    
                    result = result[tuple(indices)]
                    
                current_shape = result.shape
            
            return result
        else:
            # Different dimensionality - return as is or handle appropriately
            return feature