import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional


class AudioVisualizer:
    """Utility class for visualizing audio signals at different processing stages"""
    
    def __init__(self, sample_rate: int = 20000, output_dir: str = 'audio_visualizations'):
        """
        Args:
            sample_rate: Sample rate of the audio
            output_dir: Directory to save visualization images
        """
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_waveform(self, audio: np.ndarray, title: str, filename: str, 
                     stage: str = '', label: Optional[int] = None) -> str:
        """
        Plot and save audio waveform
        
        Args:
            audio: Audio signal array
            title: Title for the plot
            filename: Base filename for saving (without extension)
            stage: Processing stage description
            label: Optional label/class id
            
        Returns:
            Path to saved image
        """
        duration = len(audio) / self.sample_rate
        time = np.linspace(0, duration, len(audio))
        
        plt.figure(figsize=(12, 4))
        plt.plot(time, audio, linewidth=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        full_title = f'{title}'
        if stage:
            full_title += f' - {stage}'
        if label is not None:
            full_title += f' (Class {label})'
        
        plt.title(full_title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{filename}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def visualize_processing_stages(self, raw_audio: np.ndarray, 
                                   preprocessed_audio: np.ndarray,
                                   augmented_audio: Optional[np.ndarray] = None,
                                   filename: str = 'audio_sample',
                                   label: Optional[int] = None,
                                   augmentation_type: str = '') -> dict:
        """
        Visualize audio at all processing stages
        
        Args:
            raw_audio: Audio before preprocessing
            preprocessed_audio: Audio after preprocessing
            augmented_audio: Audio after augmentation (optional)
            filename: Base filename for saving
            label: Optional label/class id
            augmentation_type: Type of augmentation applied
            
        Returns:
            Dictionary with paths to saved images
        """
        paths = {}
        
        # Before preprocessing
        paths['before_preprocessing'] = self.plot_waveform(
            raw_audio, 
            'Audio Signal', 
            f'{filename}_1_before_preprocessing',
            'Before Preprocessing',
            label
        )
        
        # After preprocessing
        paths['after_preprocessing'] = self.plot_waveform(
            preprocessed_audio,
            'Audio Signal',
            f'{filename}_2_after_preprocessing',
            'After Preprocessing',
            label
        )
        
        # After augmentation (if provided)
        if augmented_audio is not None:
            stage_desc = f'After Augmentation ({augmentation_type})' if augmentation_type else 'After Augmentation'
            paths['after_augmentation'] = self.plot_waveform(
                augmented_audio,
                'Audio Signal',
                f'{filename}_3_after_augmentation_{augmentation_type}',
                stage_desc,
                label
            )
        
        return paths
