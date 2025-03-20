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
            # Get the collection
            reminder_collection = get_reminder_collection()
            all_reminders = reminder_collection.get()
            
            # Debug log total reminders
            reminder_count = len(all_reminders['ids']) if all_reminders else 0
            logging.info(f"Found {reminder_count} total reminders")
            
            # Process due reminders
            if all_reminders and len(all_reminders['ids']) > 0:
                for i, reminder_id in enumerate(all_reminders['ids']):
                    metadata = all_reminders['metadatas'][i]
                    reminder_text = all_reminders['documents'][i]
                    
                    # Log each reminder being checked
                    due_str = metadata.get('due_at', 'unknown')
                    chat_id_str = metadata.get('chat_id', 'unknown')
                    logging.debug(f"Checking reminder {reminder_id} for {chat_id_str}, due at {due_str}")
                    
                    # Check if this reminder is due
                    if 'due_at' in metadata:
                        try:
                            due_time = float(metadata['due_at'])
                            chat_id = int(metadata['chat_id'])
                            
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
                                
                                # Delete the reminder
                                reminder_collection.delete(ids=[reminder_id])
                                logging.info(f"Deleted reminder: {reminder_id}")
                        except Exception as e:
                            logging.error(f"Error processing reminder {reminder_id}: {e}")
            else:
                logging.info("No reminders found")
                
            # Continue with conversation analysis...

            # Analyze active conversations
            history_collection = get_history_collection()
            active_chats = set()
            
            try:
                # Get all unique chat_ids
                all_history = history_collection.get()
                
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
                    response = None  # Initialize response variable
                    
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
                                history_collection.add(
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
                
        except Exception as e:
            logging.error(f"Error retrieving reminders: {e}")
            
        # Always wait before next check
        logging.info("Waiting 30 seconds before checking reminders again")