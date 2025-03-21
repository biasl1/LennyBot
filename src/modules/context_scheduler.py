import logging
import threading
import time
import asyncio
from datetime import datetime
from typing import Tuple, Dict, List, Any, Optional
from modules.meta_context import get_meta_context
from modules.database import get_reminder_collection
from modules.ollama_service import process_message
from modules.prompts import PromptManager
import telegram
import requests

class ContextScheduler:
    """
    Enhanced scheduler that manages both reminders and context-based actions.
    """
    def __init__(self, bot=None, check_interval: int = 30):
        self.meta_context = get_meta_context()
        self.reminder_collection = get_reminder_collection()
        self.check_interval = check_interval
        self.running = False
        self.scheduler_thread = None
        self.bot = bot
        
        # Track the last time various checks were performed
        self.last_checks = {
            "reminders": 0,
            "pins": 0,
            "conversations": 0,
            "context_analysis": 0
        }
            
    def start(self):
        """Start the scheduler thread."""
        if self.running:
            return
            
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        self.meta_context.log_event("scheduler", "scheduler_started", {
            "timestamp": time.time(),
            "check_interval": self.check_interval
        })
        
        logging.info(f"Context scheduler started - will check every {self.check_interval} seconds")
    
    def _scheduler_loop(self):
        """Main scheduler loop that runs regular checks."""
        while self.running:
            current_time = time.time()
            
            try:
                # Always check reminders
                self._check_reminders(current_time)
                self.last_checks["reminders"] = current_time
                
                # Check message pins that need responses
                if current_time - self.last_checks["pins"] >= 60:  # Every minute
                    self._check_pending_pins(current_time)
                    self.last_checks["pins"] = current_time
                
                # Check ongoing conversations
                if current_time - self.last_checks["conversations"] >= 120:  # Every 2 minutes
                    self._check_ongoing_conversations(current_time)
                    self.last_checks["conversations"] = current_time
                
                # Perform deeper context analysis
                if current_time - self.last_checks["context_analysis"] >= 300:  # Every 5 minutes
                    self._analyze_context(current_time)
                    self.last_checks["context_analysis"] = current_time
                
            except Exception as e:
                logging.error(f"Error in scheduler loop: {e}", exc_info=True)
            
            # Wait until next check interval
            logging.info(f"Waiting {self.check_interval} seconds before checking again")
            time.sleep(self.check_interval)
    
    def _check_reminders(self, current_time: float):
        """Check for due reminders and send notifications."""
        logging.info(f"Checking for due reminders at {current_time}")
        
        try:
            # Get all reminders
            results = self.reminder_collection.get(
                include=["metadatas", "documents"]
            )
            
            reminder_count = len(results.get('ids', []))
            logging.info(f"Found {reminder_count} total reminders")
            
            if reminder_count == 0:
                logging.info("No reminders found")
                return
            
            # Check each reminder
            sent_count = 0
            for i, reminder_id in enumerate(results['ids']):
                metadata = results['metadatas'][i]
                message = results['documents'][i]
                
                # Skip completed reminders
                if metadata.get('completed', 'false') == 'true':
                    continue
                
                # Check if the reminder is due
                due_time = float(metadata.get('due_at', 0))
                chat_id = metadata.get('chat_id', 'unknown')
                
                if current_time >= due_time:
                    # Get the bot instance to send the reminder
                    try:
                        if self.bot is None:
                            from modules.telegram_service import get_bot
                            self.bot = get_bot()
                        
                        # Send the reminder
                        if chat_id != 'unknown' and self.bot:
                            # Ensure chat_id is an integer for Telegram API
                            asyncio.run(self.bot.send_message(
                                chat_id=int(chat_id),
                                text=f"⏰ Reminder: {message}"
                            ))
                            
                            # Mark reminder as completed
                            self.reminder_collection.update(
                                ids=[reminder_id],
                                metadatas=[{**metadata, "completed": "true"}]
                            )
                            
                            # Log the event
                            self.meta_context.log_event("scheduler", "reminder_sent", {
                                "timestamp": current_time,
                                "chat_id": chat_id,
                                "message": message,
                                "reminder_id": reminder_id
                            })
                            
                            sent_count += 1
                            logging.info(f"Sent reminder to {chat_id}: {message}")
                        else:
                            logging.warning(f"Couldn't send reminder - invalid chat_id: {chat_id}")
                    except Exception as e:
                        logging.error(f"Error sending reminder: {e}")
            
            if sent_count > 0:
                logging.info(f"Sent {sent_count} reminders")
                
        except Exception as e:
            logging.error(f"Error checking reminders: {e}")

    def _process_message_batch(self, chat_id, batch_id, messages):
        """Process a batch of messages with intelligent intent handling."""
        # Get message texts
        message_texts = [msg['message'] for msg in messages]
        combined_text = "\n".join(message_texts)
        
        # Special handling for reminder-like messages in any part of the batch
        if any("remind" in msg.lower() for msg in message_texts) or \
           any("in 10 min" in msg.lower() for msg in message_texts) or \
           any(re.search(r'in \d+ (minute|hour|day)', msg.lower()) for msg in message_texts):
            from modules.reminder_handler import process_reminder_intent, create_reminder
            
            # Process each message that might contain reminder info
            for msg in messages:
                if "remind" in msg['message'].lower() or \
                   re.search(r'in \d+ (minute|hour|day)', msg['message'].lower()):
                    user_name = msg['metadata'].get('user_name', 'User')
                    int_chat_id = int(chat_id)
                    
                    # Process the reminder intent with enhanced extraction
                    decision = process_reminder_intent(int_chat_id, user_name, msg['message'])
                    
                    # Try to create the reminder
                    success, response = create_reminder(decision)
                    
                    if success:
                        # Send via API
                        from config import Config
                        import requests
                        
                        telegram_url = f"https://api.telegram.org/bot{Config.TELEGRAM_API_TOKEN}/sendMessage"
                        payload = {
                            "chat_id": int_chat_id,
                            "text": response,
                            "parse_mode": "HTML"
                        }
                        
                        api_response = requests.post(telegram_url, json=payload)
                        if api_response.status_code == 200:
                            self.store_bot_response(chat_id, response, msg['metadata'].get('user_id', None))
                            logging.info(f"Successfully sent reminder confirmation to {chat_id}")
                    
                    # Mark the message as processed
                    self.meta_context.context_collection.update(
                        ids=[msg['pin_id']],
                        metadatas=[{**msg['metadata'], "processed": "true"}]
                    )
                    
            return
        
        # For non-reminder batches, proceed with standard batch classification 
        # First classify the overall batch intent
        from modules.decision_agent import intent_classifier
        intent, confidence = intent_classifier.classify(combined_text)
        
        # Continue with existing batched processing...

    async def _send_telegram_message(self, chat_id: int, text: str):
        """Helper function to send Telegram messages asynchronously."""
        try:
            if self.bot:
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=text
                )
            else:
                logging.error("Bot instance not available.")
        except Exception as e:
            logging.error(f"Error sending Telegram message: {e}")
            
    def _check_pending_pins(self, current_time=None):
        """Check for pins that need responses with batching by time proximity."""
        try:
            # Look for pending pins in meta-context
            pending_pins = self.meta_context.context_collection.get(
                where={"$and": [
                    {"event_type": "message_received"}, 
                    {"processed": "false"}
                ]},
                include=["metadatas", "documents"]
            )
            
            if not pending_pins or len(pending_pins.get('ids', [])) == 0:
                return
                
            # First, group all messages by chat_id
            messages_by_chat = {}
            for i, pin_id in enumerate(pending_pins['ids']):
                metadata = pending_pins['metadatas'][i]
                message = pending_pins['documents'][i]
                chat_id = metadata.get('chat_id', 'unknown')
                if chat_id == 'unknown':
                    continue
                    
                if chat_id not in messages_by_chat:
                    messages_by_chat[chat_id] = []
                    
                messages_by_chat[chat_id].append({
                    'id': pin_id,
                    'metadata': metadata,
                    'message': message
                })
            
            # For each chat, group messages by time proximity (5 minute window)
            MAX_TIME_GAP = 300  # 5 minutes in seconds
            conversation_batches = {}
            
            for chat_id, messages in messages_by_chat.items():
                batches = {}
                batch_id = 0
                
                # Sort messages by timestamp
                messages.sort(key=lambda m: int(m['metadata'].get('timestamp', 0)))
                
                # Group by time proximity
                for msg in messages:
                    timestamp = int(msg['metadata'].get('timestamp', 0))
                    
                    if not batches:
                        # First message creates first batch
                        batches[batch_id] = [msg]
                    else:
                        # Check if close to previous batch
                        last_batch = batches[max(batches.keys())]
                        last_msg = last_batch[-1]
                        last_time = int(last_msg['metadata'].get('timestamp', 0))
                        
                        if timestamp - last_time <= MAX_TIME_GAP:
                            # Add to existing batch
                            batches[max(batches.keys())].append(msg)
                        else:
                            # Create new batch
                            batch_id += 1
                            batches[batch_id] = [msg]
                
                conversation_batches[chat_id] = batches
            
            # Process each conversation batch
            total_batches_processed = 0
            
            for chat_id, batches in conversation_batches.items():
                chat_id = int(chat_id)  # Convert to int for Telegram
                
                for batch_id, messages in batches.items():
                    # Skip empty batches
                    if not messages:
                        continue
                        
                    # Get time gap between first and last message
                    if len(messages) > 1:
                        first_time = int(messages[0]['metadata'].get('timestamp', 0))
                        last_time = int(messages[-1]['metadata'].get('timestamp', 0))
                        time_gap = last_time - first_time
                    else:
                        time_gap = 0.0
                    
                    logging.info(f"Processing batch {batch_id} for chat {chat_id} with {len(messages)} messages over {time_gap}s")
                    
                    try:
                        # Get message texts
                        message_texts = [msg['message'] for msg in messages]
                        combined_text = " ".join(message_texts)
                        
                        # Classify the batch intent
                        from modules.intent_classifier import classify_intent
                        intent, confidence = classify_intent(combined_text)
                        
                        logging.info(f"Classified batch {batch_id} as intent '{intent}' with confidence {confidence}")
                        
                        # Get recent conversation context (last 10 messages)
                        context_messages = self._get_context_messages(chat_id, 10)
                        logging.info(f"Found {len(context_messages)} previous messages for context")
                        
                        # For action intents, check if there's a time component
                        time_info = None
                        if intent == 'action' or intent == 'reminder':
                            try:
                                # Import here to avoid circular imports
                                from modules.time_extractor import extract_time
                                time_info = extract_time(combined_text)
                                logging.info(f"Extracted time info: {time_info}")
                            except Exception as e:
                                logging.error(f"Error extracting time: {e}")
                                time_info = None
                        
                        # Process the message based on intent
                        if intent == 'reminder' or (intent == 'action' and time_info is not None):
                            # Handle reminder creation
                            from modules.reminder_handler import process_reminder_intent
                            
                            # Get the user name from the first message
                            user_name = messages[0]['metadata'].get('user_name', 'User')
                            
                            # Process the reminder intent
                            decision = process_reminder_intent(chat_id, user_name, combined_text)
                            
                            # Create the reminder if ready
                            if decision.get('ready_to_create', False):
                                from modules.reminder_handler import create_reminder
                                success, response = create_reminder(decision)
                                
                                if success:
                                    # Mark messages as processed
                                    for msg in messages:
                                        self.meta_context.context_collection.update(
                                            ids=[msg['id']],
                                            metadatas=[{**msg['metadata'], "processed": "true"}]
                                        )
                                    
                                    # Send the confirmation message
                                    if self.bot:
                                        try:
                                            asyncio.run(self._send_telegram_message(chat_id, response))
                                            self.store_bot_response(chat_id, response)
                                        except Exception as e:
                                            logging.error(f"Error sending confirmation: {e}")
                                            
                                    total_batches_processed += 1
                        else:
                            # For other intents, use ollama
                            context_text = "\n".join([f"{msg['role']}: {msg['text']}" for msg in context_messages])
                            
                            # Create prompt with context
                            prompt = f"Previous messages:\n{context_text}\n\nUser: {combined_text}"
                            
                            # Log the prompt being sent to Ollama
                            logging.info(f"Sending prompt to Ollama: {prompt[:100]}...")
                            
                            # Get response from Ollama
                            response = process_message(prompt)
                            
                            # Log the generated response
                            logging.info(f"Generated response for batch {batch_id}: {response[:50]}...")
                            
                            # Send the response
                            if self.bot:
                                try:
                                    asyncio.run(self._send_telegram_message(chat_id, response))
                                    self.store_bot_response(chat_id, response)
                                    logging.info(f"Successfully sent message to {chat_id} for batch {batch_id}")
                                except Exception as e:
                                    logging.error(f"Error sending message: {e}")
                                    
                            # Mark messages as processed
                            for msg in messages:
                                self.meta_context.context_collection.update(
                                    ids=[msg['id']],
                                    metadatas=[{**msg['metadata'], "processed": "true"}]
                                )
                                
                            total_batches_processed += 1
                    
                    except Exception as e:
                        logging.error(f"Error processing batch {batch_id} for chat {chat_id}: {e}")
                
            # Log summary
            num_chats = len(conversation_batches)
            total_msgs = sum(len(msg) for chat_batches in conversation_batches.values() for msg in chat_batches.values())
            total_batches = sum(len(batches) for batches in conversation_batches.values())
            
            if total_msgs > 0:
                logging.info(f"Found {total_msgs} pending messages in {total_batches} conversation batches across {num_chats} chats")
                logging.info(f"Successfully processed {total_batches_processed}/{total_batches} conversation batches")
            
        except Exception as e:
            logging.error(f"Error checking pending pins: {e}", exc_info=True)
    
    def store_bot_response(self, chat_id, response_text, user_id=None):
        """Store bot's response in the context system."""
        timestamp = int(time.time())
        pin_id = f"telegram-message_sent-{timestamp}"
        
        # Store the bot's response with proper metadata
        self.meta_context.context_collection.add(
            ids=[pin_id],
            documents=[response_text],  # Store the actual response text
            metadatas=[{
                "event_type": "message_sent",
                "user_id": "bot",
                "user_name": "LennyBot",
                "chat_id": str(chat_id),
                "timestamp": timestamp,
                "processed": "true",
                "source": "telegram",
                "content_type": "text",
                "recipient_id": user_id if user_id else ""
            }]
        )
        
        logging.info(f"Stored bot response: pin-{chat_id}-{timestamp}")

    def _check_ongoing_conversations(self, current_time: float):
        """Check for ongoing conversations that might need continuation."""
        # Get list of recent active users
        recent_messages = self.meta_context.get_context_window(
            event_types=["message_received", "message_sent"],
            minutes=30
        )
        
        # Group by chat_id to find active conversations
        conversations = {}
        for event in recent_messages:
            chat_id = event["metadata"].get("chat_id", "unknown")
            if chat_id == "unknown":
                continue
                
            timestamp = float(event["metadata"].get("timestamp", 0))
            
            if chat_id not in conversations:
                conversations[chat_id] = {
                    "last_message": timestamp,
                    "message_count": 1,
                    "events": [event]
                }
            else:
                conversations[chat_id]["message_count"] += 1
                if timestamp > conversations[chat_id]["last_message"]:
                    conversations[chat_id]["last_message"] = timestamp
                conversations[chat_id]["events"].append(event)
        
        # Analyze each conversation
        for chat_id, data in conversations.items():
            # Skip if last message was very recent (< 5 min) or too old (> 20 min)
            time_since_last = current_time - data["last_message"]
            if time_since_last < 300 or time_since_last > 1200:
                continue
            
            # If there are multiple messages but no recent activity
            if data["message_count"] >= 3 and 300 < time_since_last < 1200:
                logging.info(f"Found paused conversation for chat {chat_id}, inactive for {int(time_since_last)}s")
                
                # Log this event
                self.meta_context.log_event("scheduler", "paused_conversation_detected", {
                    "timestamp": current_time,
                    "chat_id": chat_id,
                    "inactive_seconds": time_since_last,
                    "message_count": data["message_count"]
                })
    
    def _analyze_context(self, current_time: float):
        """Perform deeper analysis of context for insights."""
        # Get list of active users in the last hour
        active_users = set()
        recent_events = self.meta_context.get_context_window(
            minutes=60,
            event_types=["message_received"]
        )
        
        for event in recent_events:
            chat_id = event["metadata"].get("chat_id", "unknown")
            if chat_id != "unknown":
                active_users.add(chat_id)
        
        # Analyze context for each active user
        for chat_id in active_users:
            # Get unified context
            context = self.meta_context.get_unified_context(chat_id)
            
            # Log the analysis
            self.meta_context.log_event("scheduler", "context_analysis", {
                "timestamp": current_time,
                "chat_id": chat_id,
                "context_length": len(context)
            })
    
    def stop(self):
        """Stop the scheduler."""
        self.running = False
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=1.0)
        logging.info("Context scheduler stopped")

    def _get_context_messages(self, chat_id, limit=10):
        """Get recent conversation messages for context."""
        try:
            # Get recent messages for this chat
            events = self.meta_context.get_context_window(
                chat_id=chat_id,
                event_types=["message_received", "message_sent"],
                limit=limit
            )
            
            # Format messages for context
            context_messages = []
            for event in events:
                metadata = event["metadata"]
                is_user = metadata.get("event_type") == "message_received"
                
                context_messages.append({
                    "role": "user" if is_user else "bot",
                    "text": event["data"],
                    "timestamp": float(metadata.get("timestamp", 0))
                })
            
            # Sort by timestamp
            context_messages.sort(key=lambda x: x["timestamp"])
            
            # Return the most recent messages up to the limit
            return context_messages[-limit:]
        
        except Exception as e:
            logging.error(f"Error getting context messages: {e}")
            return []

# Create a singleton accessor
_context_scheduler_instance = None

def get_context_scheduler(bot=None) -> ContextScheduler:
    global _context_scheduler_instance
    if _context_scheduler_instance is None:
        _context_scheduler_instance = ContextScheduler(bot)
    return _context_scheduler_instance

# Add bot accessor in telegram_service.py
def get_bot():
    """Get the Telegram bot instance."""
    from modules.telegram_service import application
    return application.bot if application else None