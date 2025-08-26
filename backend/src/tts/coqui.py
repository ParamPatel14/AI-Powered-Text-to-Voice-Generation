import asyncio
import numpy as np
from typing import List, Optional, Union, Dict, Any
from pathlib import Path
import tempfile
import torch

from TTS.api import TTS
from .engine_base import TTSEngineBase

class CoquiXTTSEngine(TTSEngineBase):
    """Coqui XTTS-v2 engine implementation"""
    
    SUPPORTED_LANGUAGES = [
        "en", "es", "fr", "de", "it", "pt", "pl", "tr", 
        "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"
    ]
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model_path", "tts_models/multilingual/multi-dataset/xtts_v2")
        self.device = self._get_device()
        
    def _get_device(self) -> str:
        """Determine the best available device"""
        device = self.config.get("device", "auto")
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    async def load_model(self) -> None:
        """Load the XTTS-v2 model"""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, 
                lambda: TTS(self.model_name).to(self.device)
            )
            self.is_loaded = True
            print(f"✅ Loaded Coqui XTTS model on {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to load Coqui XTTS model: {e}")
            raise
    
    async def synthesize(
        self, 
        text: str, 
        language: str = "en",
        speaker_wav: Optional[Union[str, Path, np.ndarray]] = None,
        **kwargs
    ) -> np.ndarray:
        """Synthesize speech using XTTS-v2"""
        
        if not self.is_loaded:
            await self.load_model()
            
        # Validate language
        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Language '{language}' not supported. Available: {self.SUPPORTED_LANGUAGES}")
        
        # Validate text length
        max_length = kwargs.get("max_length", 500)
        if len(text) > max_length:
            raise ValueError(f"Text too long. Maximum {max_length} characters allowed.")
        
        try:
            loop = asyncio.get_event_loop()
            
            # Handle speaker reference audio
            speaker_wav_path = None
            if speaker_wav is not None:
                if isinstance(speaker_wav, (str, Path)):
                    speaker_wav_path = str(speaker_wav)
                else:
                    # Save numpy array to temp file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        import soundfile as sf
                        sf.write(f.name, speaker_wav, 22050)
                        speaker_wav_path = f.name
            
            # Synthesize in thread pool
            if speaker_wav_path:
                # Voice cloning mode
                audio = await loop.run_in_executor(
                    None,
                    lambda: self.model.tts(
                        text=text,
                        speaker_wav=speaker_wav_path,
                        language=language
                    )
                )
            else:
                # Default speaker mode
                audio = await loop.run_in_executor(
                    None,
                    lambda: self.model.tts(
                        text=text,
                        language=language
                    )
                )
            
            # Clean up temp file
            if speaker_wav_path and isinstance(speaker_wav, np.ndarray):
                Path(speaker_wav_path).unlink(missing_ok=True)
            
            return np.array(audio, dtype=np.float32)
            
        except Exception as e:
            print(f"❌ Synthesis failed: {e}")
            raise
    
    async def get_supported_languages(self) -> List[str]:
        """Get supported language codes"""
        return self.SUPPORTED_LANGUAGES.copy()
