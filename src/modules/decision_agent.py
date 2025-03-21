import time
import logging
import torch
import re 
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from modules import ollama_service
from typing import Dict, List, Any
import os
from modules.user_interaction import update_conversation_state, get_conversation_state

# Add to imports
from modules.meta_context import get_meta_context
from modules.time_extractor import extract_time, get_current_time_formatted
from modules.database import get_reminder_collection

# Initialize the model and tokenizer
MODEL_NAME = "distilbert-base-uncased"
INTENTS = ["chat", "question", "reminder", "action", "search"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/intent_classifier")

class IntentClassifier:
    def __init__(self, model_name=MODEL_NAME, use_cached=True):
        """Initialize the DistilBERT-based intent classifier."""
        logging.info(f"Initializing intent classifier with {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set up paths
        self.model_path = MODEL_PATH
        os.makedirs(self.model_path, exist_ok=True)
        
        try:
            if use_cached and os.path.exists(self.model_path):
                logging.info(f"Loading model from cached path: {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
                self.model.to(self.device)
                logging.info("Intent classifier initialized successfully")
            else:
                logging.info(f"Downloading model from HuggingFace: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = DistilBertForSequenceClassification.from_pretrained(
                    model_name, 
                    num_labels=len(INTENTS)
                )
                self.model.to(self.device)
                
                # Save the model for future use
                self.tokenizer.save_pretrained(self.model_path)
                self.model.save_pretrained(self.model_path)
                logging.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logging.error(f"Error loading intent model: {e}")
            self.model = None
            self.tokenizer = None
            logging.info("Falling back to rule-based classification")

    def classify(self, text):
        """Classify the intent of a text message with enhanced reminder detection."""
        text_lower = text.lower()
        
        # First check for explicit reminder patterns
        if any(word in text_lower for word in ["remind", "reminder", "remember", "don't forget"]):
            # Check for time indicators
            if any(word in text_lower for word in ["in", "at", "tomorrow", "tonight", "later", "minute", "hour", "day"]):
                return "reminder", 0.95
        
        # Use the model-based prediction for other cases
        predictions = self.predict(text)
        intent = predictions["intent"]
        confidence = predictions["confidence"]
        return intent, confidence
    def predict(self, text):
        """Classify the input text with enhanced reminder detection."""
        # First check for explicit reminder patterns (already implemented)
        text_lower = text.lower()
        
        # Direct reminder patterns check...
        
        # Use the model for other classification
        if self.model is None or self.tokenizer is None:
            result = self._rule_based_classification(text)
            return {"intent": result, "confidence": 0.7}
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
            confidence, predicted_class = torch.max(probabilities, dim=0)
            intent = INTENTS[predicted_class.item()]
            confidence_value = confidence.item()
            
            logging.info(f"Classified intent: {intent} with confidence {confidence_value:.2f}")
            return {"intent": intent, "confidence": confidence_value}
        except Exception as e:
            logging.error(f"Error in intent classification: {e}")
            result = self._rule_based_classification(text)
            return {"intent": result, "confidence": 0.6}
            
    def _rule_based_classification(self, text: str) -> str:
        """Fallback rule-based classification method."""
        text = text.lower()
        
        if any(word in text for word in ["remind", "remember", "don't forget"]):
            return "reminder"
        
        if any(q in text for q in ["?", "what", "how", "why", "when", "where", "who"]):
            return "question"
            
        if any(word in text for word in ["search", "find", "lookup", "look up"]):
            return "search"
            
        if any(word in text for word in ["do", "execute", "run", "start", "stop", "create"]):
            return "action"
            
        return "chat"

# Create a global instance
intent_classifier = IntentClassifier()

def extract_reminder_details(message: str) -> Dict[str, Any]:
    """Extract reminder text and time from a message."""
    timestamp = time.time()
    
    # Check for common time patterns
    time_pattern = r'\b(in|at|after|later|tomorrow|tonight|next|this)\b|\d+\s*(minute|min|hour|day|week|month|am|pm)'
    has_time = bool(re.search(time_pattern, message.lower()))
    
    reminder_details = {
        "has_time": has_time,
        "ready_to_create": False
    }
    
    # If time information exists, extract it
    if has_time:
        due_time, time_str = extract_time(message, timestamp)
        reminder_details["due_time"] = due_time
        reminder_details["time_str"] = time_str
        
        # Extract the reminder message by removing time and command parts
        content = re.sub(r'\bremind me\b|\breminder\b|\bin\b.*|\bat\b.*|\bafter\b.*|\blater\b.*|\btomorrow\b.*|\btonight\b.*', '', message, flags=re.IGNORECASE)
        content = re.sub(r'\bto\b', '', content, 1, flags=re.IGNORECASE).strip()
        
        # Clean up the content
        if content and len(content) > 2:
            reminder_details["reminder_text"] = content
            reminder_details["ready_to_create"] = True
        else:
            # Try harder to extract content
            content = re.sub(r'remind me to ', '', message, flags=re.IGNORECASE)
            content = re.sub(r'\bin\b.*|\bat\b.*|\bafter\b.*|\blater\b.*|\btomorrow\b.*|\btonight\b.*', '', content, flags=re.IGNORECASE).strip()
            if content and len(content) > 2:
                reminder_details["reminder_text"] = content
                reminder_details["ready_to_create"] = True
    
    return reminder_details

def create_reminder(chat_id: int, user_name: str, reminder_details: Dict[str, Any]) -> str:
    """Create a reminder in the database and return confirmation message."""
    reminder_collection = get_reminder_collection()
    meta_context = get_meta_context()
    timestamp = time.time()
    
    if not reminder_details.get("ready_to_create", False):
        return "I couldn't understand when to remind you. Please specify a time like 'in 10 minutes' or 'tomorrow at 3pm'."
    
    reminder_text = reminder_details.get("reminder_text", "")
    due_time = reminder_details.get("due_time", 0)
    time_str = reminder_details.get("time_str", "")
    
    # Generate unique reminder ID
    reminder_id = f"reminder-{chat_id}-{int(timestamp)}"
    
    try:
        # Store the reminder in ChromaDB
        reminder_collection.add(
            documents=[reminder_text],
            metadatas=[{
                "chat_id": str(chat_id),
                "user_name": user_name,
                "created_at": str(timestamp),
                "due_at": str(due_time),
                "time_str": time_str,
                "completed": "false"
            }],
            ids=[reminder_id]
        )
        
        # Log the reminder creation to meta-context
        meta_context.log_event("reminder", "reminder_created", {
            "timestamp": timestamp,
            "chat_id": chat_id,
            "reminder_id": reminder_id,
            "reminder_text": reminder_text,
            "due_at": due_time,
            "time_str": time_str
        })
        
        # Format the confirmation message
        import datetime
        due_dt = datetime.datetime.fromtimestamp(due_time)
        if "at" in time_str:
            time_display = f"at {due_dt.strftime('%I:%M %p')}"
        else:
            time_display = time_str
            
        return f"✅ I'll remind you about '{reminder_text}' {time_display}."
    
    except Exception as e:
        logging.error(f"Error creating reminder: {e}")
        return "I had trouble setting up that reminder. Please try again."

def snowball_prompt(message: str, context: List[Dict] = None, chat_id: int = None, user_name: str = None) -> Dict[str, Any]:
    """Enhanced snowball prompt with meta-context integration."""
    meta_context = get_meta_context()
    
    # Step 1: Classify the intent using DistilBERT
    intent, confidence = intent_classifier.classify(message)
    
    # Log intent classification to meta-context
    meta_context.log_event("intent", "intent_classified", {
        "timestamp": time.time(),
        "chat_id": chat_id, 
        "user_name": user_name,
        "message": message,
        "intent": intent,
        "confidence": confidence
    })
    
    # Get unified context
    if chat_id:
        context_str = meta_context.get_unified_context(chat_id)
    else:
        context_str = context if isinstance(context, str) else "\n".join([str(c) for c in context]) if context else ""
    
    # Initialize action details
    action_details = {}
    
    # Enhanced specialized processing based on intent type
    if intent == "question":
        # Time-specific handling
        if any(word in message.lower() for word in ["time", "date", "day", "month", "year", "hour"]):
            current_time = get_current_time_formatted()
            action_details["current_time"] = current_time
            action_details["is_time_question"] = True
            
            # Special system prompt for time questions
            system_prompt = "You are LennyBot. This is a time-related question. Answer with the current time information."
            
            # Generate specialized time response
            prompt = f"""
            The user wants to know about time: "{message}"
            
            The current time is: {current_time}
            
            Create a friendly, concise response that includes this time information.
            """
            
            response_plan = ollama_service.process_message(prompt, system_prompt=system_prompt)
            return {
                "intent": intent,
                "confidence": confidence,
                "original_message": message,
                "action_details": action_details,
                "response_plan": response_plan,
                "user_id": chat_id,
                "user_name": user_name
            }
    
    elif intent == "reminder" or (intent == "action" and any(word in message.lower() for word in ["remind", "reminder", "remember"])):
        # Enhanced reminder handling with immediate creation
        reminder_details = extract_reminder_details(message)
        action_details.update(reminder_details)
        
        # Check for list reminders request
        if re.search(r'\b(list|show|all|my)\s+reminders\b', message.lower()):
            action_details["list_reminders"] = True
            # Skip reminder creation, just return intent for listing
        elif reminder_details.get("ready_to_create", False):
            # Create the reminder directly
            confirmation_message = create_reminder(chat_id, user_name, reminder_details)
            
            # Return result with confirmation
            return {
                "intent": "reminder",
                "confidence": confidence,
                "original_message": message,
                "action_details": action_details,
                "response_plan": confirmation_message,
                "user_id": chat_id,
                "user_name": user_name,
                "reminder_created": True
            }
        
    elif intent == "search":
        # Search enhancement - extract key search terms
        action_details["search_terms"] = [term for term in re.findall(r'\b\w{3,}\b', message.lower()) 
                                         if term not in ["the", "and", "search", "for", "find", "about", "information"]]
        
        # Try to fetch from knowledge store for relevant context
        try:
            from modules.knowledge_store import KnowledgeStore
            knowledge_store = KnowledgeStore()
            search_query = " ".join(action_details["search_terms"][:5])  # Use top 5 terms
            results = knowledge_store.search_knowledge(search_query, limit=2)
            if results:
                action_details["knowledge_results"] = results
                knowledge_text = "\n".join([result["content"] for result in results])
                
                # Add knowledge context 
                context_str += f"\n\nRELEVANT KNOWLEDGE:\n{knowledge_text}\n"
        except Exception as e:
            logging.error(f"Error fetching from knowledge store: {e}")
            
    # For non-reminder intents, generate the intent analysis with the specialized context
    if intent != "reminder" or not action_details.get("ready_to_create", False):
        try:
            intent_analysis = ollama_service.send_to_ollama(
                prompt=f"""Analyze this user message: '{message}'
                
                CONTEXT:
                {context_str[:2000]}
                
                Identify the core intent and extract any specific details needed to fulfill this request.
                Focus on: time references, entities, actions, and any missing information.
                """,
                system_prompt=f"You are a specialized intent analyzer for {intent} requests. Extract key information that would be needed to fulfill this type of request."
            )
            
            # Log the analysis
            meta_context.log_event("intent", "intent_analyzed", {
                "timestamp": time.time(),
                "chat_id": chat_id,
                "intent": intent,
                "analysis_summary": intent_analysis.get("response", "")[:100]
            })
            
            # Create specialized system prompt based on intent
            system_prompt = f"You are LennyBot, a helpful assistant. You're responding to a {intent} request."
            
            if intent == "question" and action_details.get("is_time_question"):
                system_prompt += f"\nInclude the current time ({action_details['current_time']}) in your response."
            
            if intent == "search" and action_details.get("knowledge_results"):
                system_prompt += "\nIncorporate the knowledge search results in your response."
            
            # Generate response plan
            prompt = f"""
            Based on:
            - User's message: '{message}'
            - Intent: {intent} (confidence: {confidence:.2f})
            - Analysis: {intent_analysis.get('response', '')[:300]}
            
            CONVERSATION CONTEXT:
            {context_str[:500]}
            
            Create a helpful, concise response that directly addresses the user's request.
            For questions, provide accurate information. For chat, be engaging but brief.
            """
            
            # Only generate a response for non-reminder intents or list_reminders
            response_plan = ""
            if intent != "reminder" or action_details.get("list_reminders"):
                response_plan = ollama_service.process_message(prompt, system_prompt=system_prompt)
                
            # Update action_details for the response
            action_details["response_plan"] = response_plan
        except Exception as e:
            logging.error(f"Error in intent analysis: {e}")
    
    # Return comprehensive action dictionary
    return {
        "intent": intent,
        "confidence": confidence,
        "original_message": message,
        "action_details": action_details,
        "response_plan": action_details.get("response_plan", ""),
        "user_id": chat_id,
        "user_name": user_name
    }


def decide_action(pin: dict) -> dict:
    """Enhanced decision agent using the snowball prompt system."""
    message = pin["message"]
    chat_id = int(pin.get("chat_id", "0"))
    user_name = pin.get("user_name", "User")
    
    # Get current conversation state
    current_state = get_conversation_state(chat_id)
    
    # Handle continuing conversations
    if current_state and current_state.get("current_intent") and (time.time() - current_state.get("last_update", 0) < 180):
        current_intent = current_state.get("current_intent")
        turn_count = current_state.get("turns", 0)
        
        logging.info(f"Continuing {current_intent} conversation, turn {turn_count}")
        
        # For ongoing reminder conversations, maintain the reminder intent
        if current_intent == "reminder":
            from modules.reminder_handler import process_reminder_intent
            return process_reminder_intent(chat_id, user_name, message)
    
    # For new conversations, use the snowball prompt system
    try:
        # Get recent conversation context
        from modules.meta_context import get_meta_context
        context = get_meta_context().get_unified_context(chat_id, minutes=10)
        
        # Use the snowball prompt system with proper parameters
        result = snowball_prompt(message, context=context, chat_id=chat_id, user_name=user_name)
        
        # Update conversation state based on intent
        if result["intent"] != "chat":
            update_conversation_state(chat_id, result["intent"], increment_turn=True)
        
        return result
    except Exception as e:
        logging.error(f"Error in snowball prompt: {e}")
        
        # Fallback to simple intent classification
        intent, confidence = intent_classifier.classify(message)
        logging.info(f"Classified intent (fallback): {intent} with confidence {confidence}")
        
        if intent == "reminder":
            from modules.reminder_handler import process_reminder_intent
            return process_reminder_intent(chat_id, user_name, message)
        
        return {
            "intent": intent,
            "confidence": confidence,
            "original_message": message,
            "user_id": chat_id,
            "user_name": user_name
        }

def get_classifier_info():
    """Return information about the intent classifier for self-awareness."""
    return {
        "model_name": "distilbert-base-uncased fine-tuned",
        "intents": INTENTS,  # Use the global INTENTS list for consistency
        "threshold": 0.55,  # Current confidence threshold
        "loaded": intent_classifier is not None and intent_classifier.model is not None
    }