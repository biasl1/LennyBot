import re
import datetime
import logging
import time
from telegram import Update
from telegram.ext import ContextTypes
from modules import ollama_service
from modules.conversation_context import get_time_window_context
from modules.user_interaction import update_conversation_state, get_conversation_state
from modules.database import get_history_collection, get_reminder_collection
from modules.time_extractor import extract_time

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
        
        # Get recent conversation context (last hour)
        try:
            recent_context = get_time_window_context(chat_id, minutes=10)
        except Exception as e:
            logging.error(f"Error retrieving context: {e}")
            recent_context = "No recent context available."
        
        # Get conversation state
        current_state = get_conversation_state(chat_id)
        

        # Replace the reminder handling section
        if intent == "reminder":
            # Extract time using the new module
            due_time, time_str = extract_time(user_message, timestamp)
            
            # Format for display
            due_dt = datetime.datetime.fromtimestamp(due_time)
            logging.info(f"Setting reminder for: {time_str} ({due_dt})")
            
            # Generate a unique ID
            reminder_id = f"reminder-{chat_id}-{int(timestamp)}"
            
            # Clean reminder message - remove time references
            clean_message = re.sub(r'remind me|reminder|at \d{1,2}(?::\d{2})?(?:\s*(?:am|pm))?|in \d+ (?:minute|min|hour|hr)s?', '', user_message, flags=re.IGNORECASE)
            clean_message = clean_message.strip()
            if len(clean_message) < 3:
                clean_message = user_message  # Use original if too short
            
            # Store reminder
            reminder_collection.add(
                documents=[clean_message],
                metadatas=[{
                    "chat_id": str(chat_id),
                    "user_name": user_name,
                    "created_at": str(timestamp),
                    "message": clean_message,
                    "due_at": str(due_time),
                    "time_str": time_str
                }],
                ids=[reminder_id]
            )
            
            # Send confirmation
            time_phrase = f"at {due_dt.strftime('%I:%M %p')}" if "at" in time_str else time_str
            confirmation = f"✅ I'll remind you about that {time_phrase}."
            await update.message.reply_text(confirmation)
            
            # Store bot's response
            try:
                history_collection.add(
                    documents=[confirmation],
                    metadatas=[{
                        "chat_id": str(chat_id),
                        "timestamp": str(time.time()),
                        "is_user": "false"
                    }],
                    ids=[f"reply-{unique_id}-conf"]
                )
            except Exception as e:
                logging.error(f"Error storing confirmation: {e}")
            
            return
            
        elif "my name" in user_message.lower():
            # Special handling for name questions
            response = f"Your name is {user_name}."
            await update.message.reply_text(response)
            
            # Store bot's response
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
                
            return
            
        elif intent == "question":
            prompt = f"""
            You are LennyBot, an AI assistant talking to {user_name}.
            
            RECENT CONTEXT (from the last hour):
            {recent_context}
            
            Answer this question briefly: '{user_message}'
            Keep your answer under 40 words.
            Never say you are OpenAI, ChatGPT, or any other AI. You are LennyBot.
            """
            
        else:  # chat or default
            prompt = f"""
            You are LennyBot, an AI assistant talking to {user_name}.
            
            RECENT CONTEXT (from the last hour):
            {recent_context}
            
            Reply briefly to: '{user_message}'
            Keep your reply under 30 words.
            Never say you are OpenAI, ChatGPT, or any other AI. You are LennyBot.
            """
        
        # Build an enhanced prompt with self-awareness
        prompt = f"""You are LennyBot, a friendly and helpful Telegram assistant.

CONVERSATION CONTEXT:
{recent_context}

SYSTEM AWARENESS:
- Current intent: {action['intent']}
- Conversation turns: {current_state.get('turns', 1)}
- Confidence level: {action.get('confidence', 'unknown')}

USER MESSAGE: {action['original_message']}

Based on this context and system state, provide a helpful response. If the conversation has multiple turns, ensure continuity.
"""
        
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