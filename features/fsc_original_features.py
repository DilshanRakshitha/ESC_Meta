import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import pickle
import random
import audiomentations as aa
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import yaml
import warnings

warnings.filterwarnings('ignore')


class FSCAugmentor:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.audio_config = config['data_processing']['audio']
        self.aug_config = config['data_processing']['augmentation']
        
        self.input_length = self.audio_config['input_length']
        self.sample_rate = self.audio_config['sample_rate']
        
        noise_config = self.aug_config['gaussian_noise']
        self.gaussian_noise = aa.AddGaussianNoise(
            min_amplitude=noise_config['min_amplitude'],
            max_amplitude=noise_config['max_amplitude'],
            p=noise_config['probability']
        )
        
        print(f"FSC Feature Augmentor initialized")
        print(f"Input length: {self.input_length}")
        print(f"Sample rate: {self.sample_rate}")
    
    def random_crop(self, sound: np.ndarray, size: int) -> np.ndarray:
        org_size = len(sound)
        if org_size <= size:
            return sound
        start = random.randint(0, org_size - size)
        return sound[start: start + size]
    
    def padding(self, sound: np.ndarray, size: int) -> np.ndarray:
        diff = size - len(sound)
        if diff <= 0:
            return sound
        return np.pad(sound, (diff//2, diff-(diff//2)), 'constant')
    
    def augmentor(self, audio: np.ndarray, augmentation_type: int) -> np.ndarray:
        """
        Args:
            audio: Input audio array
            augmentation_type: 
                1 = Original (no augmentation)
                2 = Time stretch fast (1.5x) rate is defined in config for all
                3 = Time stretch slow (0.667x) 
                4 = Pitch shift up (+2 semitones)
                5 = Pitch shift down (-2 semitones)
                6 = Gaussian noise + reverse
        """
        if augmentation_type == 1:
            return audio
        elif augmentation_type == 2:
            rate = self.aug_config['time_stretch']['fast_rate']
            spedup_sound = librosa.effects.time_stretch(y=audio, rate=rate)
            return self.padding(spedup_sound, self.input_length)
        elif augmentation_type == 3:
            rate = self.aug_config['time_stretch']['slow_rate']
            slowed_sound = librosa.effects.time_stretch(y=audio, rate=rate)
            return self.random_crop(slowed_sound, self.input_length)
        elif augmentation_type == 4:
            n_steps = self.aug_config['pitch_shift']['up_steps']
            return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
        elif augmentation_type == 5:
            n_steps = self.aug_config['pitch_shift']['down_steps']
            return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
        elif augmentation_type == 6:
            # Reverse + Gaussian noise
            reversed_audio = audio[::-1]
            return self.gaussian_noise(reversed_audio, sample_rate=self.sample_rate)
        else:
            return audio


class FSCFeatureExtractor:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.audio_config = config['data_processing']['audio']
        self.feature_config = config['data_processing']['features']
        
        self.sample_rate = self.audio_config['sample_rate']
        self.feature_type = self.feature_config['type']
        
        print(f"Feature Extractor initialized")
        print(f"Feature type: {self.feature_type}")
        print(f"Sample rate: {self.sample_rate}")
    
    def mel_features_extractor(self, raw_audio: np.ndarray) -> np.ndarray:
        """
        MEL spectrogram extraction with multi-scale analysis
        Uses 3 different FFT window sizes for comprehensive frequency analysis
        """
        mel_config = self.feature_config['mel']
        n_mels = mel_config['n_mels']
        n_fft_variants = mel_config['n_fft_variants']
        hop_length = mel_config['hop_length']
        
        # Multi-scale MEL spectrograms
        feature_1 = librosa.power_to_db(
            librosa.feature.melspectrogram(
                y=raw_audio, sr=self.sample_rate, n_mels=n_mels, 
                n_fft=n_fft_variants[0], hop_length=hop_length
            )
        )
        
        feature_2 = librosa.power_to_db(
            librosa.feature.melspectrogram(
                y=raw_audio, sr=self.sample_rate, n_mels=n_mels,
                n_fft=n_fft_variants[1], hop_length=hop_length
            )
        )
        
        feature_3 = librosa.power_to_db(
            librosa.feature.melspectrogram(
                y=raw_audio, sr=self.sample_rate, n_mels=n_mels,
                n_fft=n_fft_variants[2], hop_length=hop_length
            )
        )
        
        # Stack as 3-channel image (H, W, C)
        three_channel = np.stack((feature_1, feature_2, feature_3), axis=2)
        return three_channel
    
    def mfcc_features_extractor(self, raw_audio: np.ndarray) -> np.ndarray:
        """
        MFCC extraction with multi-scale analysis
        """
        mfcc_config = self.feature_config['mfcc']
        n_mfcc = mfcc_config['n_mfcc']
        n_fft_variants = mfcc_config['n_fft_variants']
        hop_length = mfcc_config['hop_length']
        
        # Multi-scale MFCC features
        feature_1 = librosa.feature.mfcc(
            y=raw_audio, sr=self.sample_rate, n_mfcc=n_mfcc,
            n_fft=n_fft_variants[0], hop_length=hop_length
        )
        
        feature_2 = librosa.feature.mfcc(
            y=raw_audio, sr=self.sample_rate, n_mfcc=n_mfcc,
            n_fft=n_fft_variants[1], hop_length=hop_length
        )
        
        feature_3 = librosa.feature.mfcc(
            y=raw_audio, sr=self.sample_rate, n_mfcc=n_mfcc,
            n_fft=n_fft_variants[2], hop_length=hop_length
        )
        
        # Stack as 3-channel image (H, W, C)
        three_channel = np.stack((feature_1, feature_2, feature_3), axis=2)
        return three_channel
    
    def mixed_features_extractor(self, raw_audio: np.ndarray) -> np.ndarray:
        """
        Mixed features (MEL + MFCC combination)
        """
        mixed_config = self.feature_config['mixed']
        mel_weight = mixed_config['mel_weight']
        mfcc_weight = mixed_config['mfcc_weight']
        
        # Extract both feature types
        mel_features = self.mel_features_extractor(raw_audio)
        mfcc_features = self.mfcc_features_extractor(raw_audio)
        
        # Weighted combination
        mixed_features = (mel_weight * mel_features + mfcc_weight * mfcc_features)
        return mixed_features
    
    def extract_features(self, raw_audio: np.ndarray) -> np.ndarray:
        
        if self.feature_type == 'mel':
            return self.mel_features_extractor(raw_audio)
        elif self.feature_type == 'mfcc':
            return self.mfcc_features_extractor(raw_audio)
        elif self.feature_type == 'mixed':
            return self.mixed_features_extractor(raw_audio)
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")


class FSCDataLoader:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_config = config['data']
        self.audio_config = config['data_processing']['audio']
        self.aug_config = config['data_processing']['augmentation']
        
        self.augmentor = FSCAugmentor(config)
        self.feature_extractor = FSCFeatureExtractor(config)
    
    def load_fsc_pickle_data(self, feature_type: str = None) -> Tuple[np.ndarray, np.ndarray]:
        
        if feature_type is None:
            feature_type = self.data_config['fsc_original']['default_feature_type']
        
        print(f"Loading FSC preprocessed data: {feature_type}")
        
        base_path = Path(self.data_config['fsc_original']['base_path'])
        pickle_dir = base_path / feature_type
        
        if not pickle_dir.exists():
            raise FileNotFoundError(f"FSC pickle directory not found: {pickle_dir}")
        
        train_spects = []
        
        for fold in range(5):
            fold_file = pickle_dir / f"{feature_type}_fold{fold+1}"
            
            if fold_file.exists():
                with open(fold_file, 'rb') as f:
                    fold_data = pickle.load(f)
                train_spects.extend(fold_data)
        
        if not train_spects:
            raise ValueError("No FSC Original data loaded!")
        
        # Convert to features and labels
        train_features_df = pd.DataFrame(train_spects, columns=['feature', 'class'])
        X = np.array(train_features_df['feature'].tolist())
        y = np.array(train_features_df['class'].tolist())
        
        # Convert from (N,H,W,C) to (N,C,H,W) for PyTorch
        if len(X.shape) == 4 and X.shape[-1] == 3:
            X = np.transpose(X, (0, 3, 1, 2))
        
        # Ensure 0-based labels
        if y.min() > 0:
            y = y - 1
        
        print(f"Final shape: {X.shape}")
        print(f"Labels: {len(np.unique(y))} classes ({y.min()} to {y.max()})")
        
        return X.astype(np.float32), y.astype(np.int64)
    
    def load_raw_audio_with_fsc_processing(self, csv_path: str, audio_path: str, 
                                         max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        
        print("Loading raw audio with FSC processing...")
        
        df = pd.read_csv(csv_path)
        if max_samples:
            df = df.head(max_samples)
        
        all_features = []
        all_labels = []
        
        duration = self.audio_config['duration']
        sample_rate = self.audio_config['sample_rate']
        input_length = self.audio_config['input_length']
        
        print(f"Processing {len(df)} audio files with FSC pipeline...")
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"   Progress: {idx}/{len(df)} ({100*idx/len(df):.1f}%)")
            
            audio_file = os.path.join(audio_path, row['filename'])
            if not os.path.exists(audio_file):
                continue
            
            try:
                # Load audio
                audio, _ = librosa.load(audio_file, sr=sample_rate, duration=duration)
                
                # Ensure consistent length
                if len(audio) < input_length:
                    audio = self.augmentor.padding(audio, input_length)
                elif len(audio) > input_length:
                    audio = self.augmentor.random_crop(audio, input_length)
                
                # Apply FSC augmentation strategies
                if self.aug_config['enabled']:
                    augmentation_types = [1, 2, 3, 4, 5, 6]
                    
                    for aug_type in augmentation_types:
                        augmented_audio = self.augmentor.augmentor(audio, aug_type)
                        
                        features = self.feature_extractor.extract_features(augmented_audio)
                        
                        if len(features.shape) == 3:  # (H, W, C) -> (C, H, W)
                            features = np.transpose(features, (2, 0, 1))
                        
                        all_features.append(features)
                        all_labels.append(row['target'] - 1)  # 0-based
                else:
                    features = self.feature_extractor.extract_features(audio)
                    if len(features.shape) == 3:
                        features = np.transpose(features, (2, 0, 1))
                    
                    all_features.append(features)
                    all_labels.append(row['target'] - 1)
                    
            except Exception as e:
                continue
        
        features = np.array(all_features, dtype=np.float32)
        labels = np.array(all_labels, dtype=np.int64)
        
        print(f"Processed: {len(features)} samples")
        print(f"Feature shape: {features.shape}")
        print(f"Augmentation factor: {len(features) / len(df):.1f}x")

        return features, labels
    
    def load_data(self, strategy: str = 'auto', **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            strategy: 'auto', 'fsc_original', 'raw_audio'
        """
        if strategy == 'auto':
            
            try:
                return self.load_fsc_pickle_data(**kwargs)
            except Exception as e:
                print(f"FSC feature extracted data not available: {e}")
                print("Falling back to raw audio processing...")
                csv_path = kwargs.get('csv_path', self.data_config['raw_audio']['csv_path'])
                audio_path = kwargs.get('audio_path', self.data_config['raw_audio']['audio_path'])
                return self.load_raw_audio_with_fsc_processing(csv_path, audio_path, **kwargs)
        
        elif strategy == 'fsc_original':
            return self.load_fsc_pickle_data(**kwargs)
        
        elif strategy == 'raw_audio':
            csv_path = kwargs.get('csv_path', self.data_config['raw_audio']['csv_path'])
            audio_path = kwargs.get('audio_path', self.data_config['raw_audio']['audio_path'])
            return self.load_raw_audio_with_fsc_processing(csv_path, audio_path, **kwargs)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


def load_config(config_path: str) -> Dict[str, Any]:
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
