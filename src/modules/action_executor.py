# ADD THIS AT THE VERY TOP - BEFORE ANY OTHER IMPORTS
from typing import Tuple, Dict, List, Any, Optional, Union

import re
import datetime
import logging
import time
import uuid
from telegram import Update
from telegram.ext import ContextTypes
from modules import ollama_service
from typing import Tuple, Dict, List, Any, Optional
from modules.user_interaction import update_conversation_state, get_conversation_state
from modules.database import get_history_collection, get_reminder_collection
from modules.time_extractor import extract_time
from modules.meta_context import get_meta_context, enhance_with_knowledge
from modules.prompts import PromptManager
from modules.knowledge_store import KnowledgeStore

async def execute_action(update: Update, context: ContextTypes.DEFAULT_TYPE, action: dict):
    """Execute an action based on user intent and context."""
    # Extract basic information
    chat_id = update.effective_chat.id
    user_name = update.effective_user.first_name
    user_message = action.get("original_message", "")
    intent = action.get("intent", "chat")
    timestamp = time.time()
    
    # Get meta-context
    meta_context = get_meta_context()
    
    # Log action execution start
    meta_context.log_event("action", "action_execution_started", {
        "timestamp": timestamp,
        "chat_id": chat_id,
        "intent": intent,
        "user_name": user_name
    })
    
    # Generate unique ID for this interaction
    unique_id = f"pin-{chat_id}-{int(timestamp)}"
    logging.info(f"Stored pin: {unique_id}")
    
    # Store this message in conversation history
    history_collection = get_history_collection()
    reminder_collection = get_reminder_collection()
    
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
    
    try:
        # Get recent conversation context
        context_str = meta_context.get_unified_context(chat_id, minutes=10)
        
        # Handle time-related questions
        current_time = None
        if intent == "question" and any(word in user_message.lower() 
                                    for word in ["time", "clock", "hour", "date", "day", "today"]):
            current_time = get_current_time_formatted()
            
            # Log specialized response
            meta_context.log_event("action", "time_response", {
                "timestamp": time.time(),
                "chat_id": chat_id,
                "time_provided": current_time
            })
            
            # If snowball provided a response_plan with time placeholder, update it
            if action.get("response_plan") and ("[insert" in action.get("response_plan", "") or 
                                                "current time" in action.get("response_plan", "")):
                action["response_plan"] = f"It's currently {current_time}."
        
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
            # Get knowledge enhancement if applicable
            knowledge = None
            try:
                # Extract key terms and search for knowledge
                knowledge_store = KnowledgeStore()
                search_terms = [term for term in re.findall(r'\b\w{3,}\b', user_message.lower()) 
                             if term not in ["the", "and", "for", "that", "this", "with", "you", "what", "how", "when"]]
                
                if search_terms:
                    search_query = " ".join(search_terms[:5])
                    results = knowledge_store.search_knowledge(search_query, limit=1)
                    
                    if results and len(results) > 0:
                        knowledge = results[0].get("content", "")
                        
                        # Log the knowledge enhancement
                        meta_context.log_event("action", "knowledge_enhanced", {
                            "timestamp": time.time(),
                            "chat_id": chat_id,
                            "search_terms": search_terms[:5],
                            "knowledge_id": results[0].get("id", "unknown")
                        })
            except Exception as e:
                logging.error(f"Error retrieving knowledge: {e}")
            
            # Create a prompt using the PromptManager
            prompt = PromptManager.create_action_prompt(
                message=user_message,
                intent=intent,
                context=context_str,
                turns=current_state.get('turns', 1) if current_state else 1,
                confidence=action.get('confidence', 0.0),
                knowledge=knowledge,
                time=current_time
            )
            
            # Log the prompt (debug level)
            logging.debug(f"Prompt for {intent}: {prompt[:100]}...")
            
            # Get response using the appropriate system prompt for this intent
            response = ollama_service.process_message(
                message=prompt, 
                system_role=intent if intent in PromptManager.SYSTEM_PROMPTS else "general"
            )
        
        # Apply post-processing to clean up the response
        response = PromptManager.post_process_response(response)
        
        # Final validation - ensure we have text to send
        if not response or len(response.strip()) == 0:
            response = PromptManager.get_fallback_response(intent)
            if user_name:
                response = response.replace("{user_name}", user_name)
        
        # Send response
        await update.message.reply_text(response)
        logging.info(f"Response sent: {response[:30]}...")
        
        # Log the response to meta-context
        meta_context.log_event("action", "message_sent", {
            "timestamp": time.time(),
            "chat_id": chat_id,
            "message": response[:100],  # Log first 100 chars
            "intent": intent
        })
        
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
        # Log the error to meta-context
        meta_context.log_event("action", "action_execution_error", {
            "timestamp": time.time(),
            "chat_id": chat_id,
            "error": str(e)
        })
        
        logging.error(f"Error in execute_action: {e}", exc_info=True)
        
        # Use standardized error fallback response
        await update.message.reply_text(PromptManager.get_fallback_response("error"))