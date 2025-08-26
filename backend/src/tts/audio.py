import numpy as np
import librosa
import soundfile as sf
from typing import Union, Tuple
from pathlib import Path
import io

class AudioProcessor:
    """Audio processing utilities"""
    
    @staticmethod
    def load_audio(file_path: Union[str, Path], target_sr: int = 22050) -> np.ndarray:
        """Load audio file and resample if needed"""
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return audio.astype(np.float32)
    
    @staticmethod
    def save_audio(audio: np.ndarray, file_path: Union[str, Path], sample_rate: int = 22050) -> None:
        """Save audio array to file"""
        sf.write(file_path, audio, sample_rate)
    
    @staticmethod
    def audio_to_bytes(audio: np.ndarray, sample_rate: int = 22050, format: str = "wav") -> bytes:
        """Convert audio array to bytes"""
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format=format.upper())
        buffer.seek(0)
        return buffer.read()
    
    @staticmethod
    def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """Normalize audio to target peak level"""
        if np.max(np.abs(audio)) == 0:
            return audio
        return audio * (target_peak / np.max(np.abs(audio)))
    
    @staticmethod
    def trim_silence(audio: np.ndarray, top_db: int = 20) -> np.ndarray:
        """Trim silence from audio"""
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return audio_trimmed
