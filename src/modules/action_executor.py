import re
import datetime
import logging
import time
from telegram import Update
from telegram.ext import ContextTypes
from modules import ollama_service
from modules.conversation_context import get_time_window_context, enhance_with_knowledge
from modules.user_interaction import update_conversation_state, get_conversation_state
from modules.database import get_history_collection, get_reminder_collection
from modules.time_extractor import extract_time
# Add missing import
import uuid

async def execute_action(update: Update, context: ContextTypes.DEFAULT_TYPE, action: dict):
    # Get collections from database module
    history_collection = get_history_collection()
    reminder_collection = get_reminder_collection()
    
    try:
        # Extract basic information
        intent = action.get("intent", "chat")
        user_message = action.get("original_message", "")
        user_name = update.effective_user.first_name
        chat_id = update.effective_chat.id
        timestamp = time.time()
        
        # Generate unique ID for this interaction
        unique_id = f"pin-{chat_id}-{int(timestamp)}"
        logging.info(f"Stored pin: {unique_id}")
        
        # Store this message in conversation history
        try:
            history_collection.add(
                documents=[user_message],
                metadatas=[{
                    "chat_id": str(chat_id),
                    "user_name": user_name,
                    "timestamp": str(timestamp),
                    "is_user": "true"
                }],
                ids=[unique_id]
            )
        except Exception as e:
            logging.error(f"Error storing message in history: {e}")
        
        # Send typing indicator
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        
        # Get conversation state
        current_state = get_conversation_state(chat_id)
        
        # Handle intent-specific actions
        if intent == "reminder":
            # Check if this is a special reminder request like "list my reminders"
            if re.search(r'(list|show|view|do i have|any) reminders', user_message.lower()):
                # Get all active reminders for this user
                results = reminder_collection.get(
                    where={"chat_id": str(chat_id), "completed": "false"},
                    include=["metadatas", "documents"]
                )
                
                # Show active reminders
                reminder_count = len(results.get('ids', []))
                if reminder_count > 0:
                    reminders_text = "Your active reminders:\n\n"
                    for i, reminder_id in enumerate(results['ids']):
                        metadata = results['metadatas'][i]
                        due_time = float(metadata.get('due_at', 0))
                        due_dt = datetime.datetime.fromtimestamp(due_time)
                        time_str = metadata.get('time_str', '')
                        message = results['documents'][i]
                        
                        if "at" in time_str:
                            time_display = f"at {due_dt.strftime('%I:%M %p')}"
                        else:
                            time_display = time_str
                            
                        reminders_text += f"• {message} - {time_display}\n"
                    
                    await update.message.reply_text(reminders_text)
                else:
                    await update.message.reply_text("You don't have any active reminders.")
                
                # Store bot's response
                try:
                    history_collection.add(
                        documents=[reminders_text if reminder_count > 0 else "You don't have any active reminders."],
                        metadatas=[{
                            "chat_id": str(chat_id),
                            "timestamp": str(time.time()),
                            "is_user": "false"
                        }],
                        ids=[f"reply-{uuid.uuid4()}"]
                    )
                except Exception as e:
                    logging.error(f"Error storing response: {e}")
                
                return
            
            # Process normal reminder creation
            from modules.reminder_handler import create_reminder
            
            # Get action details from snowball prompt if available
            action_details = action.get("action_details", {})
            
            # Use the appropriate action dictionary
            reminder_action = action_details if action_details else action
            
            # Create the reminder
            success, message = create_reminder(reminder_action)
            
            # Send message to user
            await update.message.reply_text(message)
            
            # Store bot's response in history
            try:
                history_collection.add(
                    documents=[message],
                    metadatas=[{
                        "chat_id": str(chat_id),
                        "timestamp": str(time.time()),
                        "is_user": "false"
                    }],
                    ids=[f"reply-{uuid.uuid4()}"]
                )
            except Exception as e:
                logging.error(f"Error storing response: {e}")
            
            return
        
        # For other intents, use the response plan from snowball if available
        response_plan = action.get("response_plan")
        
        if response_plan and len(response_plan.strip()) > 0:
            # Use the pre-generated response from snowball
            response = response_plan
        else:
            # Fallback to traditional method
            # Get recent conversation context
            try:
                recent_context = get_time_window_context(chat_id, minutes=10)
            except Exception as e:
                logging.error(f"Error retrieving context: {e}")
                recent_context = "No recent context available."
            
            # Build an enhanced prompt with self-awareness and knowledge
            prompt = f"""You are LennyBot, a friendly and helpful Telegram assistant.

CONVERSATION CONTEXT:
{recent_context}

SYSTEM AWARENESS:
- Current intent: {action['intent']}
- Conversation turns: {current_state.get('turns', 1) if current_state else 1}
- Confidence level: {action.get('confidence', 'unknown')}

USER MESSAGE: {action['original_message']}

Based on this context and system state, provide a helpful response. If the conversation has multiple turns, ensure continuity.
"""
            
            # Enhance with knowledge
            prompt = enhance_with_knowledge(prompt, chat_id)
            
            # Get response with safeguards
            response = ollama_service.process_message(prompt)
        
        # Final validation - ensure we have text to send
        if not response or len(response.strip()) == 0:
            response = f"I understand. How else can I help you, {user_name}?"
        
        # Send response
        await update.message.reply_text(response)
        logging.info(f"Response sent: {response[:30]}...")
        
        # Store bot's response in conversation history
        try:
            history_collection.add(
                documents=[response],
                metadatas=[{
                    "chat_id": str(chat_id),
                    "timestamp": str(time.time()),
                    "is_user": "false"
                }],
                ids=[f"reply-{unique_id}"]
            )
        except Exception as e:
            logging.error(f"Error storing bot response in history: {e}")
        
    except Exception as e:
        logging.error(f"Error in execute_action: {e}", exc_info=True)
        # Always send a valid fallback
        await update.message.reply_text("I'm having trouble processing that. Let me know if you'd like to try again.")

    # Get the context
    recent_context = get_time_window_context(chat_id, minutes=10)
    
    # Log the context (for debugging)
    logging.debug(f"CONTEXT WINDOW for {chat_id}:\n{recent_context}")