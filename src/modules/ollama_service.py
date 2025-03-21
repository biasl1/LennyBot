import requests
import logging
import time
from config import Config

def process_message(message: str, max_retries=2, system_prompt=None) -> str:
    """Optimized Ollama API wrapper with strict constraints to prevent hallucinations."""
    start_time = time.time()
    
    for attempt in range(max_retries):
        try:
            # Pass the system_prompt parameter to send_to_ollama
            response = send_to_ollama(message, system_prompt=system_prompt)
            logging.info(f"Ollama response time: {time.time() - start_time:.2f}s")
            return response.get("response", "I'm sorry, I couldn't generate a response.")
        except Exception as e:
            logging.error(f"Error in attempt {attempt + 1} with Ollama: {e}")
    
    # If we've exhausted all retries
    return "I'm having trouble right now. Please try again."

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
            timeout=180  # Increased timeout
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"Ollama API error: {response.status_code} - {response.text}")
            return {"response": "I'm having trouble connecting to my thinking module."}
            
    except Exception as e:
        logging.error(f"Error in send_to_ollama: {e}")
        return {"response": "I encountered an error while processing your request."}