import os
import logging
import re
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
import numpy as np

# Define paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "intent_classifier")

# Global variables to store the model and tokenizer
model = None
tokenizer = None
label_map = None

def init_classifier():
    """Initialize the intent classifier model."""
    global model, tokenizer, label_map
    
    if model is not None and tokenizer is not None:
        return  # Already initialized
    
    try:
        logging.info("Initializing intent classifier with distilbert-base-uncased")
        
        # Check if we have a locally cached model
        if os.path.exists(MODEL_DIR):
            logging.info(f"Loading model from cached path: {MODEL_DIR}")
            model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
            tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
        else:
            logging.info("No cached model found, loading from HuggingFace")
            model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            
            # Save model locally for future use
            os.makedirs(MODEL_DIR, exist_ok=True)
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
        
        # Define label mapping
        label_map = {
            0: "chat",
            1: "question", 
            2: "reminder",
            3: "action"
        }
        
        logging.info("Intent classifier initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing intent classifier: {e}")
        model = None
        tokenizer = None

def classify_intent(text: str) -> tuple:
    """Classify the intent of a message."""
    # Initialize if needed
    if model is None or tokenizer is None:
        init_classifier()
    
    # Simple rule-based classification as fallback
    if model is None:
        # Default to rule-based classification if model initialization failed
        return rule_based_classification(text)
    
    try:
        # First try rule-based for high-confidence cases
        rule_intent, rule_confidence = rule_based_classification(text)
        if rule_confidence > 0.9:
            return rule_intent, rule_confidence
        
        # Otherwise use the model
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().numpy()
        
        # Get the predicted label and its confidence
        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class]
        
        # Map to intent name
        intent = label_map.get(predicted_class, "chat")  # Default to chat if unknown
        
        # Blend with rule-based for hybrid approach
        if rule_confidence > confidence:
            return rule_intent, rule_confidence
        else:
            return intent, float(confidence)
    
    except Exception as e:
        logging.error(f"Error in intent classification: {e}")
        # Fallback to rule-based
        return rule_based_classification(text)

def rule_based_classification(text: str) -> tuple:
    """Simple rule-based intent classification."""
    text_lower = text.lower()
    
    # Check for reminder patterns
    reminder_patterns = [
        r'remind me',
        r'reminder',
        r'in \d+ (minutes?|hours?)',
        r'at \d{1,2}:\d{2}',
        r'at \d{1,2}(am|pm)',
        r'tomorrow',
        r'later'
    ]
    
    for pattern in reminder_patterns:
        if re.search(pattern, text_lower):
            return "reminder", 0.95
    
    # Check for question patterns
    question_patterns = [
        r'^what', 
        r'^how', 
        r'^who', 
        r'^when', 
        r'^where', 
        r'^why',
        r'^can you',
        r'^could you',
        r'\?$'
    ]
    
    for pattern in question_patterns:
        if re.search(pattern, text_lower):
            return "question", 0.9
    
    # Check for action patterns
    action_patterns = [
        r'search for',
        r'find',
        r'look up',
        r'calculate',
        r'set',
        r'do',
        r'create',
        r'send',
        r'start',
        r'stop'
    ]
    
    for pattern in action_patterns:
        if re.search(pattern, text_lower):
            return "action", 0.85
    
    # Default to chat
    return "chat", 0.7

def get_classifier_info() -> dict:
    """Return information about the current classifier."""
    return {
        "model_name": "distilbert-base-uncased",
        "intents": ["chat", "question", "reminder", "action"],
        "threshold": 0.6,
        "rule_based_fallback": True
    }

# Initialize on module load
init_classifier()