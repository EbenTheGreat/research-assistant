from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str
    PINECONE_INDEX_NAME: str
    GROQ_API_KEY: str
    GOOGLE_API_KEY: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


config = Settings()






