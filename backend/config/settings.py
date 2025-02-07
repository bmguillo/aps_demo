import os

from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
     watsonx_api_key: str = os.getenv("WATSONX_API_KEY", "")
     watsonx_url: str = os.getenv("WATSONX_URL", "")
     cors_origins: str = os.getenv("CORS_ORIGINS", "http://localhost:3000")

    
class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()