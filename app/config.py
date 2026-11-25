from typing import List, Optional, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings
    
    Reads configuration from environment variables or a .env file.
    Default values are provided for quick local development.
    """
    PROJECT_NAME: str = "Voice AI Doctor Appointment Assistant"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # Server binding configuration
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # SQLite Database connection string (creates 'appointments.db' in the project directory)
    DATABASE_URL: str = "sqlite:///./appointments.db"

    # CORS configuration: allows React frontend on localhost:3000 to communicate with this backend
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # Voice AI Service Configuration (optional so REST APIs start without voice credentials)
    GOOGLE_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-3.6-flash"
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    ELEVENLABS_API_KEY: Optional[str] = None
    ELEVENLABS_VOICE_ID: Optional[str] = None
    ELEVENLABS_STT_MODEL: str = "scribe_v2_realtime"
    ELEVENLABS_TTS_MODEL: str = "eleven_flash_v2_5"

    # Metered.ca TURN Server (optional; required for WebRTC NAT traversal on Render)
    METERED_DOMAIN: Optional[str] = None
    METERED_API_KEY: Optional[str] = None

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Union[str, List[str]]) -> List[str]:
        """
        Parses comma-separated string from .env into a list of allowed origins.
        Example: "http://localhost:3000,http://127.0.0.1:3000" -> ["http://localhost:3000", "http://127.0.0.1:3000"]
        """
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Instantiate settings to be imported throughout the project
settings = Settings()
