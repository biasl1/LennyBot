import time
import logging
import torch
import re 
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from modules import ollama_service
from typing import Dict, List, Any
import os
from modules.user_interaction import update_conversation_state, get_conversation_state

# Import the intent classifier functions correctly
from modules.intent_classifier import classify_intent, get_classifier_info


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
        """Classify the input text with enhanced reminder detection."""
        # First check for explicit reminder patterns (already implemented)
        text_lower = text.lower()
        
        # Direct reminder patterns check...
        
        # Use the model for other classification
        if self.model is None or self.tokenizer is None:
            result = self._rule_based_classification(text)
            return result, 0.7
        
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
            return intent, confidence_value
        except Exception as e:
            logging.error(f"Error in intent classification: {e}")
            result = self._rule_based_classification(text)
            return result, 0.6
            
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

def snowball_prompt(message: str, context: List[Dict] = None, chat_id: int = None, user_name: str = None) -> Dict[str, Any]:
    """
    Implement the snowball prompt system where specialized prompts are chained
    together to solve complex problems.
    """
    # Step 1: Classify the intent using DistilBERT
    intent, confidence = intent_classifier.classify(message)
    logging.info(f"Snowball classified intent: {intent} with confidence {confidence}")
    
    # Format context for prompt use
    context_str = context if isinstance(context, str) else "\n".join([str(c) for c in context]) if context else ""
    
    # Step 2: Generate specialized analysis for the intent
    intent_analysis = ollama_service.send_to_ollama(
        prompt=f"""Analyze this user message: '{message}'
        
        CONTEXT:
        {context_str[:500]}
        
        Identify the core intent and extract any specific details needed to fulfill this request.
        Focus on: time references, entities, actions, and any missing information.
        """,
        system_prompt=f"You are a specialized intent analyzer for {intent} requests. Extract key information that would be needed to fulfill this type of request."
    )
    
    # Initialize action details
    action_details = {}
    
    # Step 3: Handle specialized processing for different intents
    if intent == "reminder" and chat_id is not None:
        # Import here to avoid circular imports
        from modules.reminder_handler import process_reminder_intent
        action_details = process_reminder_intent(chat_id, user_name, message)
    elif intent == "question":
        # Question-specific analysis
        knowledge_prompt = f"""
        Given this question: '{message}'
        
        CONTEXT:
        {context_str[:300]}
        
        Extract key entities and concepts that should be researched to answer this question.
        Format as a comma-separated list.
        """
        knowledge_result = ollama_service.process_message(knowledge_prompt)
        action_details["knowledge_entities"] = knowledge_result
    
    # Step 4: Final response generation with all collected context
    system_prompt = f"""You are LennyBot, a helpful assistant.
    You're responding to a {intent} request.
    Be concise, helpful, and conversational."""
    
    prompt = f"""
    Based on:
    - User's message: '{message}'
    - Intent: {intent} (confidence: {confidence:.2f})
    - Analysis: {intent_analysis.get('response', '')[:300]}
    
    CONVERSATION CONTEXT:
    {context_str[:500]}
    
    Create a helpful, concise response that directly addresses the user's request.
    If this is a reminder, DO NOT generate a response - the system will handle it.
    For questions, provide accurate information. For chat, be engaging but brief.
    """
    
    # Only generate a response for non-reminder intents
    response_plan = ""
    if intent != "reminder":
        response_plan = ollama_service.process_message(prompt)
    
    # Return comprehensive action dictionary
    return {
        "intent": intent,
        "confidence": confidence,
        "original_message": message,
        "action_details": action_details,
        "response_plan": response_plan,
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
        from modules.conversation_context import get_time_window_context
        context = get_time_window_context(chat_id, minutes=10)
        
        # Use the snowball prompt system with proper parameters
        result = snowball_prompt(message, context=context, chat_id=chat_id, user_name=user_name)
        
        # Update conversation state based on intent
        if result["intent"] != "chat":
            update_conversation_state(chat_id, result["intent"], increment_turn=True)
        
        return result
    except Exception as e:
        logging.error(f"Error in snowball prompt: {e}")
        
        # Fallback to simple intent classification
        intent, confidence = classify_intent(message)
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
        "intents": ["chat", "question", "search", "reminder", "action"],
        "threshold": 0.35,  # Current confidence threshold
        "loaded": hasattr(get_classifier_info, "model") and get_classifier_info.model is not None
    }