import os

class Config:
    """Configuration settings for the application."""
    
    # Environment variables
    TELEGRAM_API_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    
    # Application settings
    DEBUG = os.getenv("DEBUG", "False") == "True"

    # Ollama settings
    OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi")

    # ChromaDB settings
    CHROMADB_PATH = os.getenv("CHROMADB_PATH", "./chromadb")