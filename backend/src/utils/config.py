from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Dict, List, Optional
import yaml
from pathlib import Path

class TTSEngineConfig(BaseSettings):
    name: str
    model_path: str
    device: str = "auto"
    sample_rate: int = 22050
    languages: List[str] = []
    
class APIConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_text_length: int = 1000
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    
class AppConfig(BaseSettings):
    api: APIConfig = APIConfig()
    default_engine: str = "coqui_xtts"
    model_cache_dir: str = "./models"
    temp_dir: str = "./temp"
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"

def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from file and environment variables"""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return AppConfig(**config_data)
    return AppConfig()
