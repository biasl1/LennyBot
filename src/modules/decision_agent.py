import time
import logging
import torch
import re 
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from modules import ollama_service
from typing import Dict, List, Any
import os
from modules.user_interaction import update_conversation_state, get_conversation_state

# Initialize the model and tokenizer
MODEL_NAME = "distilbert-base-uncased"
INTENTS = ["chat", "question", "reminder", "action", "search"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/intent_classifier")

class IntentClassifier:
    def __init__(self, model_name=MODEL_NAME, use_cached=True):
        """Initialize the DistilBERT-based intent classifier."""
        logging.info(f"Initializing intent classifier with {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Try to load from cached directory if it exists
            if use_cached and os.path.exists(MODEL_PATH):
                logging.info(f"Loading model from cached path: {MODEL_PATH}")
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
                self.model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
            else:
                logging.info(f"Loading model from Hugging Face: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                # Initialize with base model, would need to be fine-tuned for intents
                self.model = DistilBertForSequenceClassification.from_pretrained(
                    model_name, 
                    num_labels=len(INTENTS)
                )
            
            self.model.to(self.device)
            logging.info("Intent classifier initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing intent classifier: {str(e)}")
            # Fallback to None - will use rule-based classification if model fails
            self.model = None
            self.tokenizer = None

    def classify(self, text):
        """Classify the input text with enhanced reminder detection."""
        # First check for explicit reminder patterns
        text_lower = text.lower()
        
        # Direct reminder patterns - high confidence override
        reminder_patterns = [
            r'\bremind\b.*\bin\b',      # "remind me in 10 minutes"
            r'\bremind\b.*\bat\b',      # "remind me at 5pm"
            r'\breminder\b.*for\b',     # "set a reminder for"
            r'\bdon\'?t forget\b'       # "don't forget to"
        ]
        
        for pattern in reminder_patterns:
            if re.search(pattern, text_lower):
                logging.info(f"Reminder pattern match: {pattern}")
                return "reminder", 0.95
        
        # Use the model for other classification
        if self.model is None or self.tokenizer is None:
            result = self._rule_based_classification(text)
            return result, 0.7
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get predicted class and confidence
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]
            predicted_class = torch.argmax(probs, dim=0).item()
            confidence = probs[predicted_class].item()
            
            intent = INTENTS[predicted_class]
            
            # Boost reminder confidence if time indicators present
            if intent == "reminder" or any(word in text_lower for word in ["remind", "reminder"]):
                if any(pat in text_lower for pat in ["at", "in", "tomorrow", "later"]):
                    intent = "reminder"
                    confidence = max(confidence, 0.85)
            
            logging.info(f"Classified intent: {intent} with confidence {confidence:.2f}")
            return intent, confidence
            
        except Exception as e:
            logging.error(f"Error in intent classification: {str(e)}")
            result = self._rule_based_classification(text)
            return result, 0.7
            
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

def snowball_prompt(message: str, context: List[Dict] = None) -> Dict[str, Any]:
    """
    Implement the snowball prompt system where specialized prompts are chained
    together to solve complex problems.
    """
    # Step 1: Classify the intent
    intent = intent_classifier.classify(message)
    
    # Step 2: Generate specialized analysis for the intent
    intent_analysis = ollama_service.send_to_ollama(
        prompt=f"Analyze this user message: '{message}'\nIdentify the core intent and any specific details needed to respond appropriately.",
        system_prompt=f"You are a specialized intent analyzer for {intent} requests. Extract key information that would be needed to fulfill this type of request."
    )
    
    # Step 3: Check if any actions need to be executed
    action_analysis = ollama_service.send_to_ollama(
        prompt=f"Based on this message: '{message}'\nDetermine if any specific actions need to be taken, like setting reminders, creating tasks, or executing commands.",
        system_prompt="You are an action detector. Your job is to identify if a message implies that a specific action should be taken, and what that action should be."
    )
    
    # Step 4: Final message formulation
    final_response = ollama_service.send_to_ollama(
        prompt=f"Create a comprehensive response plan based on:\nIntent: {intent}\nIntent Analysis: {intent_analysis.get('response', '')}\nAction Analysis: {action_analysis.get('response', '')}\nOriginal Message: {message}",
        system_prompt="You are a response planner. Combine all the analysis to create a comprehensive response plan that addresses the user's needs."
    )
    
    # Construct the final decision
    return {
        "intent": intent,
        "original_message": message,
        "intent_analysis": intent_analysis.get("response", ""),
        "action_analysis": action_analysis.get("response", ""),
        "response_plan": final_response.get("response", ""),
        "needs_action": "yes" in action_analysis.get("response", "").lower()
    }


def decide_action(pin: dict) -> dict:
    """Advanced decision agent using the snowball prompt system."""
    message = pin["message"]
    chat_id = int(pin.get("chat_id", "0"))
    
    # Check current conversation state
    current_state = get_conversation_state(chat_id)
    
    # Handle continuing conversations
    if current_state and current_state.get("current_intent") and (time.time() - current_state.get("last_update", 0) < 120):
        # Continue existing conversation flow
        intent = current_state["current_intent"]
        logging.info(f"Continuing {intent} conversation, turn {current_state.get('turns', 0)}")
        
        # Update with this new message
        if current_state["current_intent"] == "reminder":
            decision = {
                "intent": "reminder",
                "original_message": message,
                "reminder_message": message,
                "user_id": chat_id
            }
            update_conversation_state(chat_id, "reminder", {"reminder_message": message})
            return decision
    
    # New conversation branch - use the snowball prompt
    decision = snowball_prompt(message)
    
    # Add the original pin data
    decision.update({
        "user_id": chat_id,
        "original_message": message
    })
    
    # Update conversation state for future messages
    update_conversation_state(chat_id, decision['intent'], {
        "original_message": message
    })
    
    logging.info(f"Decision made: {decision['intent']} for message: {message[:30]}...")
    
    return decision

def get_classifier_info():
    """Return information about the intent classifier for self-awareness."""
    return {
        "model_name": "distilbert-base-uncased fine-tuned",
        "intents": ["chat", "question", "search", "reminder", "action"],
        "threshold": 0.35,  # Current confidence threshold
        "loaded": hasattr(get_classifier_info, "model") and get_classifier_info.model is not None
    }