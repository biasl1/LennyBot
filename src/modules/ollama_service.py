import requests
import logging
import time
import json
import re
from config import Config
from modules.prompts import PromptManager

def process_message(message, system_role="general"):
    """Process a message through Ollama API using requests."""
    try:
        system_prompt = PromptManager.SYSTEM_PROMPTS.get(system_role, PromptManager.SYSTEM_PROMPTS["general"])
        
        # Force conversational style with explicit formatting
        system_prompt += """

FINAL REMINDER: 
- YOU ARE REPLYING IN A CASUAL CONVERSATION
- SPEAK DIRECTLY TO THE PERSON
- NEVER ANALYZE WHAT THEY SAID
- BE BRIEF AND FRIENDLY"""
        
        # Log shortened prompt for debugging
        logging.info(f"Sending prompt to Ollama: {message[:100]}...")
        
        start_time = time.time()
        
        # Create the payload for the Ollama API
        payload = {
            "model": "phi",  # Using phi as in the original code
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": message
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.8,
                "top_p": 0.9
            }
        }
        
        # Make the API request
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        # Process the response
        if response.status_code == 200:
            try:
                response_data = response.json()
                response_text = response_data.get('message', {}).get('content', '')
                
                if not response_text:
                    return create_contextual_fallback(message)
                    
            except json.JSONDecodeError:
                logging.warning(f"JSON decode error. Response text: {response.text[:200]}...")
                return create_contextual_fallback(message)
        else:
            logging.error(f"Ollama API error: {response.status_code}, {response.text[:200]}")
            return create_contextual_fallback(message)
        
        duration = time.time() - start_time
        
        # Log the response and time
        logging.info(f"Raw Ollama response text: {response_text[:100]}...")
        logging.info(f"Ollama response time: {duration:.2f}s")
        
        # Apply aggressive post-processing to eliminate analytical language
        clean_response = PromptManager.post_process_response(response_text)
        
        return clean_response
        
    except Exception as e:
        logging.error(f"Error in process_message: {e}")
        return create_contextual_fallback(message)

def create_contextual_fallback(message):
    """Create context-specific fallback responses instead of generic ones."""
    message_lower = message.lower()
    
    # Question patterns
    if "what is" in message_lower or "who is" in message_lower or "how do" in message_lower:
        if "dog" in message_lower:
            return "A dog is a domesticated mammal, part of the wolf family. They're known for their loyalty, friendliness, and are often kept as pets!"
        elif "cat" in message_lower:
            return "Cats are small, furry mammals that people often keep as pets. They're independent, playful, and make great companions!"
        elif "weather" in message_lower or "temperature" in message_lower:
            return "I don't have access to real-time weather data, but I'm happy to chat about other things!"
        elif "time" in message_lower:
            return "I don't have access to the current time, but I can help with other questions!"
        else:
            return "That's an interesting question! Could you tell me more about what you'd like to know?"
    
    # Greeting patterns
    elif any(word in message_lower for word in ["hello", "hi", "hey", "greetings"]):
        return "Hey there! Great to hear from you. What's on your mind today?"
    
    # Information requests
    elif any(word in message_lower for word in ["tell me", "explain", "describe"]):
        return "I'd love to explain that! Could you tell me a bit more about what you're interested in?"
    
    # Default, still conversational
    else:
        return "I'm listening! What would you like to chat about?"

def send_to_ollama(prompt, system_prompt=None):
    """
    Send a prompt to Ollama with an optional system prompt.
    Returns the full JSON response.
    """
    try:
        data = {
            "model": Config.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 100
            }
        }
        
        if system_prompt:
            data["system"] = system_prompt
            
        response = requests.post(
            f"{Config.OLLAMA_API_URL}/api/generate",
            json=data,
            timeout=360  # Increased timeout
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"Ollama API error: {response.status_code} - {response.text}")
            return {"response": create_contextual_fallback(prompt)}
            
    except Exception as e:
        logging.error(f"Error in send_to_ollama: {e}")
        return {"response": create_contextual_fallback(prompt)}