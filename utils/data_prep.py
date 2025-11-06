import random
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

def random_crop(sound, size):
    
    org_size = len(sound)
    start = random.randint(0, org_size - size)
    return sound[start: start + size]

def padding(sound, size):
    
    diff = size - len(sound)
    return np.pad(sound, (diff//2, diff-(diff//2)), 'constant')

class DataPreprocessor:
    
    
    def __init__(self, data_path: str):
        
        self.data_path = Path(data_path)
        
    def prepare_fsc22_data(self, cache: bool = True) -> Tuple[List, List]:
        
        all_folds_data = {}
        
        for fold in range(1, 6):
            fold_file = self.data_path / f"aug_ts_ps_mel_features_5_20_fold{fold}"
            
            if fold_file.exists():
                try:
                    with open(fold_file, 'rb') as f:
                        fold_data = pickle.load(f)
                    
                    all_folds_data[fold] = fold_data
                    
                except Exception as e:
                    print(f"Warning: Could not load fold {fold}: {e}")
                    continue
            else:
                print(f"Fold {fold} file not found: {fold_file}")
        
        print(f"Loaded {len(all_folds_data)} folds successfully")
        print(f"Note: Returning fold structure for proper cross-validation")
        
        return all_folds_data, None  # Return fold structure, not combined data
    
    def preprocess_features(self, features: List[np.ndarray], 
                          target_shape: Optional[Tuple] = None) -> List[np.ndarray]:

        if target_shape is None:
            return features
            
        processed = []
        for feature in features:
            
            if feature.shape != target_shape:
                
                processed.append(self._resize_feature(feature, target_shape))
            else:
                processed.append(feature)
        
        return processed
    
    def _resize_feature(self, feature: np.ndarray, target_shape: Tuple) -> np.ndarray:
        
        current_shape = feature.shape
        
        if len(current_shape) == len(target_shape):
            
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