import asyncio
import logging
import time
from telegram import Bot
from modules.conversation_context import get_time_window_context
from modules.database import get_reminder_collection, get_history_collection
from modules.ollama_service import process_message
from modules.user_interaction import get_conversation_state, update_conversation_state

class ReminderScheduler:
    def __init__(self, bot: Bot):
        self.bot = bot
        self.running = False
        self.task = None
        self.reminder_collection = get_reminder_collection()
        self.history_collection = get_history_collection()
    
    async def start(self):
        """Start the reminder checking loop."""
        self.running = True
        self.task = asyncio.create_task(self._check_reminders_loop())
        logging.info("Reminder scheduler started - will check every 30 seconds")
    
    async def stop(self):
        """Stop the reminder checking loop."""
        self.running = False
        if self.task:
            self.task.cancel()
        logging.info("Reminder scheduler stopped")
    
    async def _check_reminders_loop(self):
        """Periodically check for reminders that are due."""
        while self.running:
            try:
                await self._process_due_reminders()
            except Exception as e:
                logging.error(f"Error checking reminders: {e}")
            
            # Wait for 30 seconds before checking again
            logging.info("Waiting 30 seconds before checking reminders again")
            await asyncio.sleep(30)
    
    async def _process_due_reminders(self):
        """Check for and send due reminders."""
        current_time = time.time()
        logging.info(f"Checking for due reminders at {current_time}")
        
        try:
            # Get all active reminders
            results = self.reminder_collection.get(
                where={"completed": "false"},
                include=["metadatas", "documents"]
            )
            
            # Debug log total reminders
            reminder_count = len(results['ids']) if results and 'ids' in results else 0
            logging.info(f"Found {reminder_count} total reminders")
            
            # Process due reminders
            if results and 'ids' in results and len(results['ids']) > 0:
                for i, reminder_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i]
                    reminder_text = results['documents'][i]
                    
                    # Check if this reminder is due
                    if 'due_at' in metadata:
                        try:
                            due_time = float(metadata['due_at'])
                            chat_id_str = metadata.get('chat_id', 'unknown')
                            
                            # Convert chat_id to int if it's a string
                            if chat_id_str and chat_id_str.isdigit():
                                chat_id = int(chat_id_str)
                            else:
                                logging.warning(f"Invalid chat_id format: {chat_id_str} for reminder {reminder_id}")
                                continue
                            
                            # Log time until due for debugging
                            seconds_until = due_time - current_time
                            if seconds_until > 0:
                                logging.debug(f"Reminder {reminder_id} due in {seconds_until:.1f} seconds")
                            
                            # If reminder is due
                            if current_time >= due_time:
                                logging.info(f"⏰ REMINDER DUE! {reminder_id} for {chat_id}")
                                
                                # Send reminder with poem if requested
                                if "poem" in reminder_text.lower():
                                    prompt = f"Write a short, fun poem about: {reminder_text}. Keep it under 6 lines."
                                    poem = process_message(prompt)
                                    reminder_message = f"🔔 *REMINDER* 📝\n\n{poem}"
                                else:
                                    reminder_message = f"🔔 *REMINDER*\n\n{reminder_text}"
                                
                                await self.bot.send_message(
                                    chat_id=chat_id,
                                    text=reminder_message,
                                    parse_mode="Markdown"
                                )
                                logging.info(f"Sent reminder to {chat_id}: {reminder_text[:30]}...")
                                
                                # Mark as completed instead of deleting
                                self.reminder_collection.update(
                                    ids=[reminder_id],
                                    metadatas=[{"completed": "true"}]
                                )
                                logging.info(f"Marked reminder {reminder_id} as completed")
                        except Exception as e:
                            logging.error(f"Error processing reminder {reminder_id}: {e}")
            else:
                logging.info("No reminders found")
            
            # Continue with conversation analysis for proactive followups
            await self._analyze_conversations(current_time)
                
        except Exception as e:
            logging.error(f"Error retrieving reminders: {e}")
    
    async def _analyze_conversations(self, current_time):
        """Analyze active conversations and send proactive follow-ups if needed."""
        active_chats = set()
        
        try:
            # Get all unique chat_ids
            all_history = self.history_collection.get()
            
            if all_history and len(all_history['ids']) > 0:
                for metadata in all_history['metadatas']:
                    if 'chat_id' in metadata:
                        active_chats.add(metadata['chat_id'])
            
            # For each active chat, check if we need to intervene
            for chat_id_str in active_chats:
                try:
                    chat_id = int(chat_id_str)
                except ValueError:
                    logging.warning(f"Skipping invalid chat_id: {chat_id_str}")
                    continue
                
                current_state = get_conversation_state(chat_id)
                
                # Skip if no current state
                if not current_state:
                    continue
                
                # Check for incomplete conversations that need proactive follow-up
                follow_up_count = current_state.get("follow_up_count", 0)
                if (current_state.get("current_intent") in ["reminder", "action"] and 
                    current_state.get("turns", 0) == 1 and
                    current_time - current_state.get("last_update", 0) > 60 and
                    follow_up_count < 2):  # Maximum of 2 follow-ups
                    
                    # Generate follow-up prompt
                    time_window = get_time_window_context(chat_id, minutes=10)
                    
                    prompt = f"""
                    You are LennyBot, an AI assistant.
                    
                    RECENT CONVERSATION CONTEXT:
                    {time_window}
                    
                    Based on the conversation above, the user started to ask for a {current_state.get("current_intent")} 
                    but hasn't provided full details. Generate a short, helpful follow-up message to ask for the missing information.
                    Keep your response under 30 words and be conversational.
                    """
                    
                    response = process_message(prompt)
                
                    if response:
                        try:
                            await self.bot.send_message(chat_id=chat_id, text=response)
                            logging.info(f"Sent proactive follow-up for {current_state.get('current_intent')} to {chat_id}")
                            
                            # Store bot's proactive response
                            self.history_collection.add(
                                documents=[response],
                                metadatas=[{
                                    "chat_id": chat_id_str,
                                    "timestamp": str(time.time()),
                                    "is_user": "false"
                                }],
                                ids=[f"proactive-{time.time()}"]
                            )
                            
                            # Update to increment the follow-up count
                            current_state["follow_up_count"] = follow_up_count + 1
                            update_conversation_state(
                                chat_id, 
                                current_state.get("current_intent"),
                                current_state
                            )
                        except Exception as e:
                            logging.error(f"Error sending proactive message: {e}")
        
        except Exception as e:
            logging.error(f"Error in conversation analysis: {e}")
