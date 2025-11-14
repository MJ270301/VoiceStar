import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings"""

    # Basic configuration
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8001

    # Database
    DATABASE_URL: str = "sqlite:///./voice_ai_suite.db"

    # File storage
    UPLOAD_DIR: str = "./uploads"
    VOICE_MODELS_DIR: str = "./models"
    AUDIO_CACHE_DIR: str = "./cache"

    # AI Model configuration
    VOICE_MODEL_PATH: str = "../models"
    SAMPLE_RATE: int = 16000
    MAX_AUDIO_LENGTH: int = 30  # seconds

    # API keys (optional)
    ELEVENLABS_API_KEY: Optional[str] = None
    AZURE_SPEECH_KEY: Optional[str] = None
    AZURE_SPEECH_REGION: Optional[str] = None

    # TTS configuration
    DEFAULT_VOICE_PROVIDER: str = "internal"  # internal, elevenlabs, azure
    MAX_CONCURRENT_SYNTHESIS: int = 10

    # Voice cloning configuration
    MIN_SAMPLE_LENGTH: float = 5.0  # seconds
    MAX_SAMPLE_LENGTH: float = 120.0  # seconds
    SUPPORTED_FORMATS: list = ["wav", "mp3", "m4a", "flac"]

    # Emotion processing
    EMOTION_MODEL_PATH: str = "../models/emotion_detector.pt"
    EMOTION_CONFIDENCE_THRESHOLD: float = 0.7

    # Celebrity voices
    CELEBRITY_VOICES_DIR: str = "./celebrity_voices"
    LICENSE_VALIDATION: bool = True

    # Rate limiting
    VOICE_CLONE_RATE_LIMIT: int = 5  # per hour
    TTS_RATE_LIMIT: int = 100  # per minute

    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()