import asyncio
import logging
import time
from telegram import Bot
from modules.user_interaction import get_conversation_state
from modules.conversation_context import get_time_window_context
from modules.database import get_reminder_collection, get_history_collection
from modules.ollama_service import process_message

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
        """Check for and send due reminders, then analyze conversation context."""
        current_time = time.time()
        logging.info(f"Checking for due reminders at {current_time}")
        
        # Get collections from database module
        reminder_collection = get_reminder_collection()
        history_collection = get_history_collection()
        # --- PART 1: PROCESS REMINDERS (Keep existing code) ---
        try:
            # Get the collection
            reminder_collection = get_reminder_collection()
            all_reminders = reminder_collection.get()
        
            all_reminders = reminder_collection.get()
            
            if not all_reminders or not all_reminders['ids']:
                logging.info("No reminders found")
            else:
                logging.info(f"Found {len(all_reminders['ids'])} reminders to check")
                
                for i, reminder_id in enumerate(all_reminders['ids']):
                    metadata = all_reminders['metadatas'][i]
                    message = all_reminders['documents'][i]
                    
                    # Process due reminders (same as before)
                    if 'created_at' in metadata and 'chat_id' in metadata:
                        created_time = float(metadata['created_at'])
                        chat_id = metadata['chat_id']
                        
                        time_diff = current_time - created_time
                        logging.info(f"Reminder {reminder_id} age: {time_diff:.1f} seconds")
                        
                        if time_diff >= 60:  # 60 seconds = 1 minute
                            try:
                                logging.info(f"Sending reminder {reminder_id} to {chat_id}")
                                await self.bot.send_message(
                                    chat_id=int(chat_id),
                                    text=f"⏰ Reminder: {message}"
                                )
                                logging.info(f"Sent reminder {reminder_id} to {chat_id}")
                                
                                reminder_collection.delete(ids=[reminder_id])
                                logging.info(f"Deleted reminder {reminder_id}")
                                
                            except Exception as e:
                                logging.error(f"Error sending reminder {reminder_id}: {e}")
        
            # --- PART 2: NEW - ANALYZE ACTIVE CONVERSATIONS (Every 30 seconds) ---
            try:
                # Get all unique chat_ids from history collection
                active_chats = set()
                recents = history_collection.get()
                
                if recents and recents['metadatas']:
                    for metadata in recents['metadatas']:
                        if 'chat_id' in metadata:
                            try:
                                chat_id = metadata['chat_id']
                                # Only consider recent activity (last 10 minutes)
                                if 'timestamp' in metadata:
                                    ts = float(metadata['timestamp'])
                                    if current_time - ts < 600:  # 10 minutes
                                        active_chats.add(chat_id)
                            except Exception:
                                pass
                
                # For each active chat, check if we need to intervene
                for chat_id_str in active_chats:
                    try:
                        chat_id = int(chat_id_str)
                    except ValueError:
                        continue  # Skip this chat_id and proceed to the next one
                    
                    current_state = get_conversation_state(chat_id)
                    
                    # Check for incomplete conversations that need proactive follow-up
                    follow_up_count = current_state.get("follow_up_count", 0)
                    if (current_state.get("current_intent") in ["reminder", "action"] and 
                        current_state.get("turns", 0) == 1 and
                         current_time - current_state.get("last_update", 0) > 60 and
                        follow_up_count < 2):  # Maximum of 2 follow-ups
                        
                        # It's been a minute since the user requested a reminder or action
                        # but they haven't provided details - prompt them
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
                            
                            # ADD THE NEW CODE RIGHT HERE - after storing the bot's response:
                            from modules.user_interaction import update_conversation_state
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