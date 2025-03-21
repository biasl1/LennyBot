# ADD AT THE TOP - BEFORE ANY OTHER CODE
from typing import Tuple, Dict, List, Any, Optional, Union
import threading
import uuid
import ast
import logging
import time
from datetime import datetime

from modules.database import get_db_client, get_history_collection

class MetaContext:
    """
    Centralized repository for all system context and events.
    Serves as both a logging system and a context provider.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MetaContext, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """Initialize the MetaContext singleton."""
        self.db_client = get_db_client()
        
        # Create the meta_context collection
        self.context_collection = self.db_client.create_collection(
            "meta_context",
            get_or_create=True
        )
        
        # Also create or get the previously used collections for backward compatibility
        self.history_collection = self.db_client.create_collection(
            "history",
            get_or_create=True
        )
        self.reminder_collection = self.db_client.create_collection(
            "reminders",
            get_or_create=True
        )
    
    # Fix the indentation error in log_event method (around line 76-77)
    def log_event(self, source: str, event_type: str, data: Dict[str, Any], id_suffix: str = None):
        """Log an event to the meta-context repository with unique IDs."""
        try:
            timestamp = data.get("timestamp", time.time())
            
            # Generate a unique ID with random suffix
            import uuid
            random_suffix = uuid.uuid4().hex[:8]
            event_id = f"{source}-{event_type}-{int(timestamp)}-{random_suffix}"
            
            if id_suffix:
                event_id = f"{event_id}-{id_suffix}"
            
            # Add metadata
            metadata = {
                "source": source,
                "event_type": event_type,
                "timestamp": str(timestamp)
            }
            
            # Add chat_id if present in the data
            if "chat_id" in data:
                metadata["chat_id"] = str(data["chat_id"])
                
            # Store in ChromaDB
            self.context_collection.add(
                ids=[event_id],
                metadatas=[metadata],
                documents=[str(data)]
            )
            
            return event_id
        except Exception as e:
            logging.error(f"Error logging event: {e}")
            return None
    

    # Update the get_context_window function to handle complex queries properly:

    def get_context_window(self, 
                        chat_id: Optional[Union[str, int]] = None,
                        minutes: int = 10,
                        event_types: Optional[List[str]] = None,
                        source: Optional[str] = None,
                        limit: int = 50) -> List[Dict[str, Any]]:
        """Get a context window from the meta-context repository."""
        results = []
        
        try:
            # Get a larger set with simple filter (ChromaDB only supports simple where clauses)
            where_filter = {}
            if chat_id is not None:
                where_filter = {"chat_id": str(chat_id)}
            elif source is not None:
                where_filter = {"source": source}
            
            # Simple query with larger limit to allow for filtering
            events = self.context_collection.get(
                where=where_filter if where_filter else None,
                limit=limit * 3  # Get more to filter in code
            )
            
            if not events or not events.get('ids'):
                return []
            
            # Calculate time cutoff
            cutoff_time = time.time() - (minutes * 60) if minutes > 0 else 0
            
            # Filter in Python code rather than complex database queries
            for i, event_id in enumerate(events['ids']):
                metadata = events['metadatas'][i] 
                document = events['documents'][i]
                
                # Skip if it doesn't match our filters
                # 1. Time filter
                if minutes > 0:
                    try:
                        timestamp = float(metadata.get('timestamp', 0))
                        if timestamp < cutoff_time:
                            continue
                    except (ValueError, TypeError):
                        pass
                        
                # 2. Event type filter
                if event_types and metadata.get('event_type') not in event_types:
                    continue
                    
                # 3. Source filter (if not used in main query)
                if source and not where_filter.get('source') and metadata.get('source') != source:
                    continue
                    
                # Process the document
                data = document
                if isinstance(data, str):
                    try:
                        import ast
                        data = ast.literal_eval(data)
                    except (SyntaxError, ValueError):
                        pass
                        
                results.append({
                    "id": event_id,
                    "metadata": metadata,
                    "data": data
                })
                
            # Sort by timestamp
            results.sort(key=lambda x: float(x["metadata"].get("timestamp", 0)))
            
            # Apply limit
            return results[:limit]
        except Exception as e:
            logging.error(f"Error retrieving context window: {e}")
            return []
    
    def get_unified_context(self, chat_id: Union[str, int], minutes: int = 10) -> str:
        """
        Get a unified text representation of context for this chat,
        combining meta-context, message history, and system status.
        """
        # Get recent meta-context events
        meta_events = self.get_context_window(chat_id=chat_id, minutes=minutes)
        
        # Get message history from the regular history collection
        try:
            history_results = self.history_collection.get(
                where={"chat_id": str(chat_id)},
                include=["metadatas", "documents"]
            )
            
            # Filter and sort by timestamp
            messages = []
            for i, msg_id in enumerate(history_results.get('ids', [])):
                metadata = history_results['metadatas'][i]
                doc = history_results['documents'][i]
                
                if 'timestamp' in metadata:
                    try:
                        msg_time = float(metadata['timestamp'])
                        if msg_time >= (time.time() - minutes * 60):
                            is_user = metadata.get('is_user') == "true"
                            user_name = metadata.get('user_name', 'User') if is_user else 'LennyBot'
                            
                            messages.append({
                                'time': msg_time,
                                'user': user_name,
                                'is_user': is_user,
                                'text': doc
                            })
                    except ValueError:
                        continue
            
            # Sort by timestamp
            messages.sort(key=lambda x: x['time'])
            
            # Format the unified context
            context_text = "# UNIFIED CONTEXT\n\n"
            
            # Add conversation history
            if messages:
                context_text += "## Recent Conversation\n"
                for msg in messages:
                    prefix = f"{msg['user']}: " if msg['is_user'] else "LennyBot: "
                    context_text += f"{prefix}{msg['text']}\n"
                context_text += "\n"
            
            # Add system events
            if meta_events:
                context_text += "## System Events\n"
                for event in meta_events[-5:]:  # Latest 5 events
                    metadata = event["metadata"]
                    document = event["data"]
                    source = metadata.get("source", "unknown")
                    event_type = metadata.get("event_type", "unknown")
                    context_text += f"• {source}.{event_type} at {datetime.fromtimestamp(float(metadata.get('timestamp', 0))).strftime('%H:%M:%S')}: {document}\n"
                context_text += "\n"
            
            # Add user state info
            from modules.user_interaction import get_conversation_state
            state = get_conversation_state(chat_id)
            if state:
                context_text += "## Conversation State\n"
                context_text += f"• Current intent: {state.get('current_intent', 'none')}\n"
                context_text += f"• Turns: {state.get('turns', 0)}\n"
                last_update = state.get('last_update', 0)
                if last_update:
                    time_ago = time.time() - last_update
                    context_text += f"• Last update: {int(time_ago)} seconds ago\n"
            
            return context_text
            
        except Exception as e:
            logging.error(f"Error getting unified context: {e}")
            return "Error retrieving unified context."

# Create a convenient singleton accessor
def get_meta_context() -> MetaContext:
    return MetaContext()

def import_logs_to_history():
    """
    Import logs from text files into the history collection.
    This is used for migrating old conversation logs.
    """
    logging.info("Importing logs to history collection - this is a no-op in the new architecture")
    get_meta_context().log_event("system", "logs_imported", {
        "timestamp": time.time(),
        "message": "Legacy log import function called"
    })
    return

def enhance_with_knowledge(prompt, chat_id):
    """Enhance a prompt with relevant knowledge from the knowledge store."""
    try:
        # Extract key entities/topics from the prompt
        from modules.knowledge_store import KnowledgeStore
        knowledge_store = KnowledgeStore()
        
        # Simple keyword extraction (basic implementation)
        words = prompt.lower().split()
        keywords = [word for word in words if len(word) > 4 and word.isalpha()][:5]
        
        if not keywords:
            return prompt
        
        # Search for relevant knowledge
        search_query = " ".join(keywords)
        results = knowledge_store.search_knowledge(search_query, limit=2)
        
        if not results:
            return prompt
        
        # Add knowledge to the prompt
        knowledge_text = "\n\nRELEVANT KNOWLEDGE:\n"
        for i, result in enumerate(results):
            knowledge_text += f"{i+1}. {result['content']}\n"
            
        return prompt + knowledge_text
    except Exception as e:
        logging.error(f"Error enhancing with knowledge: {e}")
        return prompt