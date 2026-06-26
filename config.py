import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App Settings
    PORT: int = 8050
    HOST: str = "0.0.0.0"
    
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    COLLECTION_NAME: str = "hr_policies"
    
    LITELLM_MODEL: str = os.getenv("LITELLM_MODEL", "openai/gpt-4o-mini")
    LITELLM_API_KEY: str = os.getenv("LITELLM_API_KEY", "your-litellm-key")
    LITELLM_BASE_URL: str = os.getenv("LITELLM_BASE_URL", "")
    
    GROQ_MODEL: str = "openai/groq/llama-3.3-70b-versatile"
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "your-groq-key")
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    

    EXTERNAL_DB_BASE_URL: str = "http://localhost:8000"
    # Fallback dummy Bearer token if authorization paths are checked
    EXTERNAL_DB_TOKEN: str = os.getenv("EXTERNAL_DB_TOKEN", "mock-token")

settings = Settings()