from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any, List
import numpy as np
from pathlib import Path

class TTSEngineBase(ABC):
    """Abstract base class for TTS engines"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.is_loaded = False
        
    @abstractmethod
    async def load_model(self) -> None:
        """Load the TTS model"""
        pass
    
    @abstractmethod
    async def synthesize(
        self, 
        text: str, 
        language: str = "en",
        speaker_wav: Optional[Union[str, Path, np.ndarray]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Synthesize speech from text
        
        Args:
            text: Input text to synthesize
            language: Target language code (ISO 639-1)
            speaker_wav: Reference audio for voice cloning
            **kwargs: Additional engine-specific parameters
            
        Returns:
            Audio array (numpy array)
        """
        pass
    
    @abstractmethod
    async def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes"""
        pass
    
    async def unload_model(self) -> None:
        """Unload the model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
            self.is_loaded = False
            
    async def health_check(self) -> Dict[str, Any]:
        """Check engine health status"""
        return {
            "engine": self.__class__.__name__,
            "loaded": self.is_loaded,
            "config": self.config
        }


