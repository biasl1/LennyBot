import logging
import time
from telegram import Update
from telegram.ext import ContextTypes
from modules import ollama_service
from modules.conversation_context import get_time_window_context
from modules.user_interaction import update_conversation_state
from modules.database import get_history_collection, get_reminder_collection

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
        
        # Process based on intent
        if intent == "reminder":
            reminder_message = action.get("reminder_message", user_message)
            
            # Store reminder in ChromaDB with safe metadata
            unique_reminder_id = f"reminder-{chat_id}-{int(timestamp)}-{hash(reminder_message) % 10000}"
            reminder_data = {
                "chat_id": str(chat_id),
                "user_name": user_name,
                "created_at": str(timestamp),
                "message": reminder_message
            }
            
            # Store in ChromaDB
            try:
                reminder_collection.add(
                    documents=[reminder_message],
                    metadatas=[reminder_data],
                    ids=[unique_reminder_id]
                )
                logging.info(f"Stored reminder: {unique_reminder_id}")
            except Exception as e:
                logging.error(f"Error storing reminder: {e}")
            
            prompt = f"""
            You are LennyBot, an AI assistant.
            
            RECENT CONTEXT:
            {recent_context}
            
            The user {user_name} has asked you to remind them about: '{reminder_message}'. 
            Confirm that you will remind them in a friendly way.
            Keep your response under 30 words.
            """
            
        elif "what did we talk about" in user_message.lower() or "chat history" in user_message.lower():
            # Specific response for conversation history requests
            if recent_context and recent_context != "No recent context available.":
                response = f"Here's what we've been discussing:\n\n{recent_context}"
            else:
                response = "We haven't had much conversation recently that I can recall."
            
            await update.message.reply_text(response)
            
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