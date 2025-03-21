import logging
import threading
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

from modules.meta_context import get_meta_context
from modules.database import get_reminder_collection
from modules.ollama_service import process_message
import telegram

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
    
    def _check_pending_pins(self, current_time: float):
        """Check for message pins that need responses."""
        # Get recent pins without responses
        pins = self.meta_context.get_context_window(
            event_types=["message_received"],
            minutes=10
        )
        
        pending_count = 0
        for pin in pins:
            pin_id = pin["id"]
            pin_time = float(pin["metadata"].get("timestamp", 0))
            chat_id = pin["metadata"].get("chat_id", "unknown")
            
            # Check if this pin has a response
            responses = self.meta_context.get_context_window(
                chat_id=chat_id,
                event_types=["message_sent"],
                minutes=0  # No time limit
            )
            
            # Filter for responses that occurred after this pin
            has_response = any(
                float(r["metadata"].get("timestamp", 0)) > pin_time 
                for r in responses
            )
            
            # If no response and pin is older than 30 seconds but newer than 5 minutes
            if not has_response and (current_time - pin_time > 30) and (current_time - pin_time < 300):
                pending_count += 1
                logging.info(f"Found pending pin: {pin_id} for chat {chat_id}")
                
                # Log this event
                self.meta_context.log_event("scheduler", "pending_pin_detected", {
                    "timestamp": current_time,
                    "chat_id": chat_id,
                    "pin_id": pin_id,
                    "age_seconds": current_time - pin_time
                })
        
        if pending_count > 0:
            logging.info(f"Found {pending_count} pending pins that need responses")
    
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