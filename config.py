from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GEMINI_API_KEY: str = ""
    GCP_PROJECT:    str = "running-assistant-485215"
    GCP_REGION:     str = "europe-west1"
    DATABASE_URL:   str = ""
    VITALS_ANALYSIS_INTERVAL: int = 60
    MAX_CONVERSATION_HISTORY: int = 20

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()