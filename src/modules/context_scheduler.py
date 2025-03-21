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
            
    def _check_pending_pins(self, current_time: float):
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
                timestamp = float(metadata.get('timestamp', 0))
                
                if chat_id not in messages_by_chat:
                    messages_by_chat[chat_id] = []
                    
                messages_by_chat[chat_id].append({
                    'pin_id': pin_id,
                    'message': message,
                    'metadata': metadata,
                    'timestamp': timestamp
                })
            
            # For each chat, group messages by time proximity (5 minute window)
            MAX_TIME_GAP = 300  # 5 minutes in seconds
            conversation_batches = {}
            
            for chat_id, messages in messages_by_chat.items():
                if chat_id == 'unknown' or not chat_id.isdigit():
                    continue
                    
                # Sort messages by timestamp
                sorted_messages = sorted(messages, key=lambda x: x['timestamp'])
                
                # Group into conversation batches
                if len(sorted_messages) > 0:
                    batch_id = 0
                    conversation_batches[chat_id] = {}
                    conversation_batches[chat_id][batch_id] = [sorted_messages[0]]
                    
                    # Group subsequent messages based on time proximity
                    for i in range(1, len(sorted_messages)):
                        curr_msg = sorted_messages[i]
                        prev_msg = sorted_messages[i-1]
                        
                        # If this message is within MAX_TIME_GAP of the previous one, add to current batch
                        if curr_msg['timestamp'] - prev_msg['timestamp'] <= MAX_TIME_GAP:
                            conversation_batches[chat_id][batch_id].append(curr_msg)
                        else:
                            # Start a new batch
                            batch_id += 1
                            conversation_batches[chat_id][batch_id] = [curr_msg]
                    
                    for batch_id, batch in conversation_batches[chat_id].items():
                        for msg in batch:
                            logging.info(f"Found pending pin: {msg['pin_id']} for chat {chat_id} in batch {batch_id}")
            
            # Process each conversation batch
            total_batches_processed = 0
            
            for chat_id, batches in conversation_batches.items():
                for batch_id, messages in batches.items():
                    try:
                        # Only process if there are messages
                        if len(messages) == 0:
                            continue
                            
                        # Get message texts
                        message_texts = [msg['message'] for msg in messages]
                        
                        # Get the time range for this batch
                        start_time = min([msg['timestamp'] for msg in messages])
                        end_time = max([msg['timestamp'] for msg in messages])
                        time_range = end_time - start_time
                        
                        logging.info(f"Processing batch {batch_id} for chat {chat_id} with {len(messages)} messages over {time_range:.1f}s")
                        
                        # Intent classification
                        from modules.intent_classifier import classify_intent
                        intent, confidence = classify_intent("\n".join(message_texts))
                        logging.info(f"Classified batch {batch_id} as intent '{intent}' with confidence {confidence}")
                        
                        # Special handling for reminder intent
                        if intent == "reminder":
                            from modules.reminder_handler import process_reminder_intent, create_reminder
                            
                            # Process each message in the batch as a reminder
                            for msg in messages:
                                user_name = msg['metadata'].get('user_name', 'User')
                                int_chat_id = int(chat_id)
                                
                                # Process the reminder intent
                                decision = process_reminder_intent(int_chat_id, user_name, msg['message'])
                                
                                # Try to create the reminder
                                success, response = create_reminder(decision)
                                
                                if success:
                                    # Use direct API call for the response
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
                                    else:
                                        logging.error(f"Failed to send reminder response: {api_response.status_code} - {api_response.text}")
                                
                                # Mark the message as processed
                                self.meta_context.context_collection.update(
                                    ids=[msg['pin_id']],
                                    metadatas=[{**msg['metadata'], "processed": "true"}]
                                )
                            
                            total_batches_processed += 1
                            continue  # Skip the standard processing below
                        
                        # Add to _check_pending_pins method after intent classification
                        combined_text = "\n".join(message_texts)

                        if intent == "reminder" or "remind" in combined_text.lower():
                            # Extract time information
                            from modules.time_extractor import extract_time
                            time_info = extract_time(combined_text)
                            
                            if time_info and time_info.get('due_timestamp'):
                                # Create the reminder
                                from modules.reminder_handler import create_reminder
                                reminder_data = {
                                    "user_id": chat_id,
                                    "user_name": messages[0]['metadata'].get('user_name', 'User'),
                                    "time_info": time_info,
                                    "due_time": time_info['due_timestamp'],
                                    "reminder_message": combined_text,
                                    "timestamp": time.time()
                                }
                                success, response_message = create_reminder(reminder_data)
                                
                                # Modify the response to confirm reminder creation
                                if success:
                                    response = f"✅ {response_message}"
                                else:
                                    response = f"❌ {response_message}"

                        # Get conversation history for context
                        conversation_history = ""
                        try:
                            # Get recent history before this batch
                            history_results = self.meta_context.context_collection.get(
                                where={
                                    "$and": [
                                        {"chat_id": chat_id}, 
                                        {"timestamp": {"$lt": start_time - 10}}
                                    ]
                                },
                                include=["metadatas", "documents"],
                                limit=10
                            )

                            # Then filter event types manually:
                            filtered_results = {
                                'ids': [],
                                'metadatas': [],
                                'documents': []
                            }

                            for i, doc_id in enumerate(history_results['ids']):
                                metadata = history_results['metadatas'][i]
                                if metadata.get('event_type') in ["message_received", "message_sent"]:
                                    filtered_results['ids'].append(doc_id)
                                    filtered_results['metadatas'].append(metadata)
                                    filtered_results['documents'].append(history_results['documents'][i])
                            
                            # Format conversation history
                            if history_results and len(history_results.get('ids', [])) > 0:
                                # Create tuples of (timestamp, index) for sorting
                                timestamps = [(float(history_results['metadatas'][i].get('timestamp', 0)), i) 
                                             for i in range(len(history_results['ids']))]
                                # Sort by timestamp (descending)
                                sorted_indices = [idx for _, idx in sorted(timestamps, reverse=True)]
                                # Use sorted indices to access the results
                                sorted_messages = []
                                for idx in sorted_indices:
                                    msg_type = history_results['metadatas'][idx].get('event_type')
                                    sender = "User" if msg_type == "message_received" else "Bot"
                                    sorted_messages.append(f"{sender}: {history_results['documents'][idx]}")
                                
                                conversation_history = "\n".join(sorted_messages)
                                logging.info(f"Found {len(history_results['ids'])} previous messages for context")
                        except Exception as history_err:
                            logging.error(f"Error retrieving conversation history: {history_err}")
                        
                        # Create the prompt using PromptManager
                        if len(messages) > 1:
                            # For multiple messages, use batch template
                            prompt = PromptManager.create_batch_prompt(
                                message_texts, 
                                time_gap=time_range
                            )
                        else:
                            # For single message
                            prompt = message_texts[0]
                        
                        # Add conversation history if available
                        if conversation_history:
                            prompt = PromptManager.format_prompt(
                                "with_context",
                                context=conversation_history,
                                message=prompt
                            )
                        
                        # Get appropriate system prompt for this intent
                        system_role = intent if intent in PromptManager.SYSTEM_PROMPTS else "general"
                        
                        # Generate response with the appropriate system prompt
                        response = process_message(
                            message=prompt, 
                            system_role=system_role
                        )
                        
                        # Post-process the response
                        response = PromptManager.post_process_response(response)
                        
                        # Validate response
                        if not response or len(response.strip()) == 0:
                            response = PromptManager.get_fallback_response(intent)
                        
                        logging.info(f"Generated response for batch {batch_id}: {response[:50]}...")
                        
                        # Direct telegram API call (synchronous)
                        from config import Config
                        import requests
                        
                        # Use direct API call instead of async methods
                        telegram_url = f"https://api.telegram.org/bot{Config.TELEGRAM_API_TOKEN}/sendMessage"
                        payload = {
                            "chat_id": int(chat_id),
                            "text": response,
                            "parse_mode": "HTML"
                        }
                        
                        # Replace the API response section in _check_pending_pins with:
                        api_response = requests.post(telegram_url, json=payload)
                        if api_response.status_code == 200:
                            # Successfully sent message, now store it in context
                            user_id = messages[0]['metadata'].get('user_id', None)
                            self.store_bot_response(chat_id, response, user_id)
                            logging.info(f"Successfully sent message to {chat_id} for batch {batch_id}")
                        else:
                            logging.error(f"Failed to send message: {api_response.status_code} - {api_response.text}")
                        
                        # Mark all messages in this batch as processed
                        for msg in messages:
                            self.meta_context.context_collection.update(
                                ids=[msg['pin_id']],
                                metadatas=[{**msg['metadata'], "processed": "true"}]
                            )
                        
                        total_batches_processed += 1
                            
                    except Exception as e:
                        logging.error(f"Error processing batch {batch_id} for chat {chat_id}: {e}", exc_info=True)
                    
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