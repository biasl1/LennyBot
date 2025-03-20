import datetime
import os
import re
import logging
import time
from config import Config
from modules.database import get_history_collection, get_reminder_collection

def import_logs_to_history(log_file_path=None):
    """Import conversation history from log files."""
    if not log_file_path:
        log_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "bot.log")
    
    logging.info(f"Importing conversation history from {log_file_path}")
    
    # Get collection from database module
    history_collection = get_history_collection()
    
    imported_count = 0
    
    try:
        with open(log_file_path, "r") as file:
            for line in file:
                # Extract user or bot messages from logs
                if "Response sent:" in line or "Received message from" in line:
                    try:
                        time_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line)
                        if time_match:
                            log_time = time_match.group(1)
                        else:
                            continue
                            
                        if "Response sent:" in line:
                            # Bot message
                            content_match = re.search(r'Response sent:\s*(.*)', line)
                            if content_match:
                                message = content_match.group(1).strip()
                                is_user = False
                                chat_id = "unknown"  # Default if we can't extract
                            else:
                                continue
                        else:  # "Received message from"
                            # User message
                            name_match = re.search(r'Received message from ([^:]+):\s*(.*)', line)
                            if name_match:
                                user_name = name_match.group(1).strip()
                                message = name_match.group(2).strip()
                                is_user = True
                                
                                # Try to extract chat_id from subsequent lines
                                chat_id_match = re.search(r'pin-(\d+)-', line)
                                if chat_id_match:
                                    chat_id = chat_id_match.group(1)
                                else:
                                    chat_id = "unknown"
                            else:
                                continue
                                
                        # Store in ChromaDB
                        history_collection.add(
                            documents=[message],
                            metadatas=[{
                                "chat_id": str(chat_id),
                                "timestamp": str(time.time()),
                                "is_user": str(is_user).lower(),
                                "user_name": user_name if is_user else "LennyBot"
                            }],
                            ids=[f"import-{hash(log_time + message) % 1000000}"]
                        )
                        imported_count += 1
                    except Exception as e:
                        logging.error(f"Error importing log line: {e}")
                        
        logging.info(f"Imported {imported_count} conversation entries from logs")
    except Exception as e:
        logging.error(f"Error importing conversation logs: {e}")

def get_recent_context(chat_id, time_window_seconds=3600):
    """Get recent conversation context for a chat."""
    # Get collection from database module
    history_collection = get_history_collection()
    
    # Try to get recent conversation history
    try:
        chat_id_str = str(chat_id)
        
        # Get all messages for this chat
        history = history_collection.get(
            where={"chat_id": chat_id_str}
        )
        
        if history and len(history['ids']) > 0:
            # Sort messages by timestamp
            messages_with_time = []
            for i, msg_id in enumerate(history['ids']):
                metadata = history['metadatas'][i]
                doc = history['documents'][i]
                
                if 'timestamp' in metadata:
                    try:
                        msg_time = float(metadata['timestamp'])
                        if msg_time >= cutoff_time:
                            is_user = metadata.get('is_user') == "true"
                            user_name = metadata.get('user_name', 'User') if is_user else 'LennyBot'
                            
                            messages_with_time.append({
                                'time': msg_time,
                                'user': user_name,
                                'is_user': is_user,
                                'text': doc
                            })
                    except ValueError:
                        logging.warning(f"Invalid timestamp in metadata: {metadata.get('timestamp')}")
            
            # Sort messages by time
            messages_with_time.sort(key=lambda x: x['time'])
            
            # Format messages into a proper conversation transcript
            if messages_with_time:
                # Take the most recent N messages 
                recent_messages = messages_with_time[-10:]
                
                conversation_lines = []
                for msg in recent_messages:
                    prefix = f"{msg['user']}: " if msg['is_user'] else "LennyBot: "
                    conversation_lines.append(f"{prefix}{msg['text']}")
                
                if conversation_lines:
                    context_parts.append("Recent conversation:\n" + "\n".join(conversation_lines))
    except Exception as e:
        logging.error(f"Error retrieving conversation history: {e}")


def get_time_window_context(chat_id, minutes=10):
    """Get all context within a time window as a chronological stream."""
    context_items = []
    current_time = time.time()
    cutoff_time = current_time - (minutes * 60)
    chat_id_str = str(chat_id)
    
    # Get collections from database module
    history_collection = get_history_collection()
    reminder_collection = get_reminder_collection()
    
    try:
        # Get messages
        history = history_collection.get(
            where={"chat_id": chat_id_str}
        )
        
        if history and len(history['ids']) > 0:
            for i, msg_id in enumerate(history['ids']):
                metadata = history['metadatas'][i]
                doc = history['documents'][i]
                
                if 'timestamp' in metadata:
                    try:
                        msg_time = float(metadata['timestamp'])
                        if msg_time >= cutoff_time:
                            is_user = metadata.get('is_user') == "true"
                            user_name = metadata.get('user_name', 'User') if is_user else 'LennyBot'
                            
                            context_items.append({
                                'time': msg_time,
                                'type': 'message',
                                'user': user_name,
                                'is_user': is_user,
                                'text': doc
                            })
                    except ValueError:
                        logging.warning(f"Invalid timestamp in metadata: {metadata.get('timestamp')}")
        
        # Get reminders in the same timeframe
        reminders = reminder_collection.get(
            where={"chat_id": chat_id_str}
        )
        
        if reminders and len(reminders['ids']) > 0:
            for i, rem_id in enumerate(reminders['ids']):
                metadata = reminders['metadatas'][i]
                doc = reminders['documents'][i]
                
                if 'created_at' in metadata:
                    try:
                        rem_time = float(metadata['created_at'])
                        if rem_time >= cutoff_time:
                            context_items.append({
                                'time': rem_time,
                                'type': 'reminder',
                                'id': rem_id,
                                'text': doc
                            })
                    except ValueError:
                        pass
        
        # Sort all context items chronologically
        context_items.sort(key=lambda x: x['time'])
        
        # Format into conversational text
        if not context_items:
            return "No recent context available."
            
        # Format for the LLM
        conversation_text = "Recent conversation and activity:\n"
        for item in context_items:
            if item['type'] == 'message':
                prefix = f"{item['user']}: " if item['is_user'] else "LennyBot: "
                conversation_text += f"{prefix}{item['text']}\n"
            elif item['type'] == 'reminder':
                conversation_text += f"[System] Reminder set: {item['text']}\n"
                
        conversation_text += "\n--- System Status ---\n"
        conversation_text += f"Time window: {minutes} minutes\n"
        conversation_text += f"Current time: {datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}\n"
        conversation_text += f"Context items: {len(context_items)}\n"
        
        # Add current conversation state if available
        try:
            from modules.user_interaction import get_conversation_state
            state = get_conversation_state(chat_id)
            if state:
                conversation_text += f"Current intent: {state.get('current_intent', 'none')}\n"
                conversation_text += f"Conversation turns: {state.get('turns', 0)}\n"
                conversation_text += f"Last update: {datetime.datetime.fromtimestamp(state.get('last_update', 0)).strftime('%H:%M:%S')}\n"
        except ImportError:
            pass
            
        return conversation_text        
    except Exception as e:
        logging.error(f"Error retrieving time window context: {e}")
        return "Error retrieving context."

def enhance_with_knowledge(text, chat_id=None):
    """Enhance a message with relevant knowledge when appropriate."""
    try:
        # Don't enhance short messages
        if len(text) < 15:
            return text
            
        # Import here to avoid circular imports
        from modules.knowledge_store import KnowledgeStore
        
        # Get knowledge store
        knowledge_store = KnowledgeStore()
        
        # Search for relevant knowledge
        results = knowledge_store.search_knowledge(text, limit=1)
        
        # Only enhance if we have a good match
        if results and len(results) > 0 and results[0]['relevance'] > 0.7:
            knowledge = results[0]['content']
            topic = results[0]['topic']
            
            # Add knowledge as context
            enhanced_text = f"{text}\n\nRelevant knowledge: {knowledge} (from topic: {topic})"
            return enhanced_text
            
        return text
    except Exception as e:
        logging.error(f"Error enhancing with knowledge: {e}")
        return text