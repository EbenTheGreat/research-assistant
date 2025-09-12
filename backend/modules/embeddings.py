from langchain_google_genai import GoogleGenerativeAIEmbeddings
from backend.config.config import config

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=config.GOOGLE_API_KEY
    )
