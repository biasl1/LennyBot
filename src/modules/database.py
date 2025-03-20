import chromadb
from config import Config
from chromadb.config import Settings
import logging
import numpy as np

# Standardize all embeddings to use the same dimension
EMBEDDING_DIM = 384

def mini_embed(texts):
    """Create simple embeddings with the standardized dimensionality."""
    if isinstance(texts, str):
        texts = [texts]
        
    embeddings = []
    for text in texts:
        # Create a deterministic hash-based embedding but constrain to 32-bit range
        hash_val = hash(text) & 0xFFFFFFFF  # Constrain to 32-bit range
        np.random.seed(hash_val)
        # Create vector with consistent dimensionality
        embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        # Normalize the vector
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding.tolist())
        
    return embeddings

# Initialize ChromaDB at module level
chroma_client = chromadb.PersistentClient(
    path=Config.CHROMADB_PATH,
    settings=Settings(anonymized_telemetry=False)
)

# Create all collections
history_collection = chroma_client.get_or_create_collection(
    name="conversation_history",
    embedding_function=mini_embed
)

reminder_collection = chroma_client.get_or_create_collection(
    name="reminders",
    embedding_function=mini_embed
)

pin_collection = chroma_client.get_or_create_collection(
    name="pins",
    embedding_function=mini_embed
)

# Accessor functions
def get_history_collection():
    return history_collection

def get_reminder_collection():
    return reminder_collection

def get_pin_collection():
    return pin_collection