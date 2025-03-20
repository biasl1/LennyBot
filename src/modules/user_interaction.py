# Remove ChromaDB initialization and use database module
import logging
import datetime
import re
import time

from modules.database import get_pin_collection

# Define active conversations
active_conversations = {}  # Track active conversations by chat_id

def update_conversation_state(chat_id, intent, details=None):
    """Track the state of active conversations."""
    active_conversations[chat_id] = {
        "current_intent": intent,
        "details": details or {},
        "last_update": time.time(),
        "turns": active_conversations.get(chat_id, {}).get("turns", 0) + 1
    }
    
def get_conversation_state(chat_id):
    """Get the current conversation state."""
    state = active_conversations.get(chat_id, {})
    # Check if conversation is still active (less than 5 minutes old)
    if time.time() - state.get("last_update", 0) > 900:
        # Reset expired conversations
        if chat_id in active_conversations:
            del active_conversations[chat_id]
        return {}
    return state

def store_pin(chat_id: int, message: str) -> dict:
    """Store message as pin using lightweight embeddings."""
    # Get collection from database module
    pin_collection = get_pin_collection()
    
    # Generate a unique ID that won't collide
    unique_id = f"pin-{chat_id}-{int(time.time())}"
    
    # Extract basic entities
    keywords = re.findall(r'\b(remind|remember|question|help)\b', message, re.IGNORECASE)
    
    # Create pin object
    pin = {
        "id": unique_id,
        "chat_id": str(chat_id),
        "timestamp": datetime.datetime.now().isoformat(),
        "message": message,
        "keywords": ",".join(keywords)
    }
    
    # Store in ChromaDB
    try:
        pin_collection.add(
            documents=[message],
            metadatas=[pin],
            ids=[unique_id]
        )
        logging.info(f"Stored pin: {unique_id}")
    except Exception as e:
        logging.error(f"Error storing pin: {e}")
    
    return pin