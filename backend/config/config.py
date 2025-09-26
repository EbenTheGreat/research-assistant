from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str
    PINECONE_INDEX_NAME: str
    GROQ_API_KEY: str
    GOOGLE_API_KEY: str
    OPENAI_API_KEY: str
    VOYAGE_API_KEY: str
    GOOGLE_APPLICATION_CREDENTIAL: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


config = Settings()
