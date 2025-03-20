import time
import logging
import torch
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

    def classify(self, text: str) -> str:
        """Classify the input text into one of the predefined intents."""
        if self.model is None or self.tokenizer is None:
            return self._rule_based_classification(text)
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get predicted intent
            predicted_class = torch.argmax(outputs.logits, dim=1).item()
            intent = INTENTS[predicted_class]
            confidence = torch.softmax(outputs.logits, dim=1)[0][predicted_class].item()
            
            logging.info(f"Classified intent: {intent} with confidence {confidence:.2f}")
            
            # If confidence is low, fall back to rule-based
            if confidence < 0.35:
                logging.info(f"Low confidence ({confidence:.2f}), falling back to rule-based")
                return self._rule_based_classification(text)
                
            return intent
        except Exception as e:
            logging.error(f"Error in intent classification: {str(e)}")
            return self._rule_based_classification(text)
            
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