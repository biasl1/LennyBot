from pydantic import BaseModel

class Message(BaseModel):
    """Data model for a message."""
    chat_id: int
    text: str
    message_id: int

class TelegramMessage(BaseModel):
    """Data model for a Telegram message."""
    update_id: int
    message: Message

# Additional models can be added as needed for other message types or structures.