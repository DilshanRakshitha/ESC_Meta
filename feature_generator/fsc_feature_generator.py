import numpy as np
import librosa
import random
from typing import List, Tuple, Optional, Dict, Any
import audiomentations as aa
import warnings

warnings.filterwarnings('ignore')


class FSCFeatureGenerator:
    
    def __init__(self, sr=20000, duration=5):
        """
        Args:
            sr: Sample rate (default: 20000)
            duration: Audio duration in seconds (default: 5)
        """
        self.sr = sr
        self.duration = duration
        self.input_length = sr * duration
        
        # Audio augmentation
        self.gaussian_noise = aa.AddGaussianNoise(
                min_amplitude=0.001, 
                max_amplitude=0.015, 
                p=0.5
            )
    
    def padding(self, sound: np.ndarray, size: int) -> np.ndarray:
        diff = size - len(sound)
        return np.pad(sound, (diff//2, diff-(diff//2)), 'constant')
    
    def random_crop(self, sound: np.ndarray, size: int) -> np.ndarray:
        org_size = len(sound)
        if org_size <= size:
            return sound
        start = random.randint(0, org_size - size)
        return sound[start: start + size]
    
    def simple_gaussian_noise(self, audio: np.ndarray, noise_factor: float = 0.01) -> np.ndarray:
        noise = np.random.normal(0, noise_factor, audio.shape)
        return audio + noise
    
    def time_stretch(self, audio: np.ndarray, rate: float) -> np.ndarray:
        stretched = librosa.effects.time_stretch(y=audio, rate=rate)
        if rate > 1.0:  # Speed up - need padding
            return self.padding(stretched, self.input_length)
        else:  # Slow down - need cropping
            return self.random_crop(stretched, self.input_length)
    
    def pitch_shift(self, audio: np.ndarray, n_steps: int) -> np.ndarray:
        return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
    
    def augment_audio(self, audio: np.ndarray, augmentation_type: str) -> np.ndarray:
        """
        Args:
            audio: Input audio signal
            augmentation_type: Type of augmentation
            
        Returns:
            Augmented audio signal
        """
        if augmentation_type == 'speed_up':
            return self.time_stretch(audio, 1.5)
        elif augmentation_type == 'slow_down':
            return self.time_stretch(audio, 0.667)
        elif augmentation_type == 'pitch_up':
            return self.pitch_shift(audio, 2)
        elif augmentation_type == 'pitch_down':
            return self.pitch_shift(audio, -2)
        elif augmentation_type == 'noise':
            return self.gaussian_noise(audio, sample_rate=self.sr)
        elif augmentation_type == 'reverse_noise':
            reversed_audio = audio[::-1]
            return self.gaussian_noise(reversed_audio, sample_rate=self.sr)
        else:
            return audio
    
    def extract_mel_features(self, raw_audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Args:
            raw_audio: Input audio signal
            
        Returns:
            3-channel mel spectrogram features (H, W, C)
        """
        try:
            # Three different FFT sizes for multi-scale features - FSC Original approach
            feature_1 = librosa.power_to_db(
                librosa.feature.melspectrogram(
                    y=raw_audio, sr=self.sr, n_mels=128, 
                    n_fft=2048, hop_length=512
                )
            )
            feature_2 = librosa.power_to_db(
                librosa.feature.melspectrogram(
                    y=raw_audio, sr=self.sr, n_mels=128, 
                    n_fft=1024, hop_length=512
                )
            )
            feature_3 = librosa.power_to_db(
                librosa.feature.melspectrogram(
                    y=raw_audio, sr=self.sr, n_mels=128, 
                    n_fft=512, hop_length=512
                )
            )
            
            # Stack as 3-channel image (H, W, C) format
            three_channel = np.stack((feature_1, feature_2, feature_3), axis=2)
            return three_channel
        except Exception as e:
            print(f"Error extracting mel features: {e}")
            return None
    
    def extract_mfcc_features(self, raw_audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Args:
            raw_audio: Input audio signal
            
        Returns:
            3-channel MFCC features (H, W, C)
        """
        try:
            feature_1 = librosa.feature.mfcc(
                y=raw_audio, n_mfcc=128, n_fft=2048, hop_length=512
            )
            feature_2 = librosa.feature.mfcc(
                y=raw_audio, n_mfcc=128, n_fft=1024, hop_length=512
            )
            feature_3 = librosa.feature.mfcc(
                y=raw_audio, n_mfcc=128, n_fft=512, hop_length=512
            )
            
            three_channel = np.stack((feature_1, feature_2, feature_3), axis=2)
            return three_channel
        except Exception as e:
            print(f"Error extracting MFCC features: {e}")
            return None
    
    def extract_mixed_features(self, raw_audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Args:
            raw_audio: Input audio signal
            
        Returns:
            3-channel mixed features (H, W, C)
        """
        try:
            feature_1 = librosa.feature.mfcc(
                y=raw_audio, n_mfcc=128, n_fft=1024, hop_length=512
            )
            feature_2 = librosa.power_to_db(
                librosa.feature.melspectrogram(
                    y=raw_audio, sr=self.sr, n_mels=128, 
                    n_fft=1024, hop_length=512
                )
            )
            feature_3 = librosa.feature.chroma_stft(
                y=raw_audio, sr=self.sr, n_chroma=128, 
                n_fft=1024, hop_length=512
            )
            
            three_channel = np.stack((feature_1, feature_2, feature_3), axis=2)
            return three_channel
        except Exception as e:
            print(f"Error extracting mixed features: {e}")
            return None
    
    def extract_features_with_augmentation(self, audios: List[Tuple], feature_type: str = 'MEL', 
                                         augment_level: int = 0) -> List[List]:
        """
        Args:
            audios: List of (audio, label) tuples
            feature_type: 'MEL', 'MFCC', or 'MIX'
            augment_level: Augmentation level (0-4)
            
        Returns:
            List of [feature, label, is_original] triplets
        """
        print(f"Extracting {feature_type} features with augmentation level {augment_level}")
        
        if feature_type == 'MEL':
            extractor = self.extract_mel_features
        elif feature_type == 'MFCC':
            extractor = self.extract_mfcc_features
        elif feature_type == 'MIX':
            extractor = self.extract_mixed_features
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        features = []
        
        for audio, label in audios:
            try:
                # Normalize audio length
                if len(audio) < self.input_length:
                    audio = self.padding(audio, self.input_length)
                elif len(audio) > self.input_length:
                    audio = self.random_crop(audio, self.input_length)
                
                # Original sample
                original_feature = extractor(audio)
                if original_feature is not None:
                    features.append([original_feature, label, True])  # True = original
                
                # Apply augmentation based on level
                augmentation_types = []
                if augment_level == 1:  # Time stretch only
                    augmentation_types.extend(['speed_up', 'slow_down'])
                if augment_level == 2:  # Pitch shift only
                    augmentation_types.extend(['pitch_up', 'pitch_down'])
                if augment_level >= 3:  # Both time stretch and pitch shift
                    augmentation_types.extend(['speed_up', 'slow_down'])
                    augmentation_types.extend(['pitch_up', 'pitch_down'])
                if augment_level >= 4:  # All augmentations including noise
                    augmentation_types.extend(['noise', 'reverse_noise'])
                
                # Generate augmented samples
                for aug_type in augmentation_types:
                    aug_audio = self.augment_audio(audio, aug_type)
                    aug_feature = extractor(aug_audio)
                    if aug_feature is not None:
                        features.append([aug_feature, label, False])  # False = augmented
            
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
        
        print(f"Extracted {len(features)} feature samples total")
        return features
    
    def prepare_features_for_training(self, features: List[List]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            features: List of [feature, label, is_original] triplets
            
        Returns:
            Tuple of (X, y) numpy arrays
        """
        X = np.array([f[0] for f in features])
        y = np.array([f[1] for f in features])
        
        print(f"Prepared features shape: {X.shape}")
        print(f"Prepared labels shape: {y.shape}")
        
        return X, y
