"""Configuration settings for the Multimodal Agentic RAG system."""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    OPENAI_API_KEY: str
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    PDF_DIR: Path = DATA_DIR / "pdfs"
    IMAGE_DIR: Path = DATA_DIR / "images"
    TEXT_DIR: Path = DATA_DIR / "texts"
    TABLE_DIR: Path = DATA_DIR / "tables"
    FAISS_INDEX_DIR: Path = DATA_DIR / "faiss_index"
    
    # Model Configuration
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    OPENAI_VISION_MODEL: str = "gpt-4-vision-preview"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    VISION_MODEL: str = "llava-hf/llava-1.5-7b-hf"
    
    # Chunking Configuration
    CHUNK_SIZE: int = 4000
    CHUNK_OVERLAP: int = 200
    MAX_CHARS: int = 4000
    NEW_AFTER_N_CHARS: int = 3800
    COMBINE_UNDER_N_CHARS: int = 2000
    
    # Retrieval Configuration
    TOP_K_RETRIEVAL: int = 4
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Agent Configuration
    AGENT_TEMPERATURE: float = 0.7
    MAX_ITERATIONS: int = 5
    
    # Memory Configuration
    MEMORY_WINDOW: int = 10  # Number of previous messages to keep
    
    # Reinforcement Learning
    RL_LEARNING_RATE: float = 0.001
    RL_GAMMA: float = 0.99
    RL_REWARD_THRESHOLD: float = 0.8
    
    # MCP Configuration
    MCP_SERVER_URL: Optional[str] = None
    MCP_ENABLED: bool = False
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        for dir_path in [self.PDF_DIR, self.IMAGE_DIR, self.TEXT_DIR, self.TABLE_DIR, self.FAISS_INDEX_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()