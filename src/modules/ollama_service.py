import requests
import logging
import time
from config import Config

def process_message(message: str, max_retries=2) -> str:
    """Optimized Ollama API wrapper with strict constraints to prevent hallucinations."""
    start_time = time.time()
    
    for attempt in range(max_retries):
        try:
            timeout = 60 + (attempt * 5)  # 20s, then 25s
            
            # Format prompt to clearly distinguish context from current message
            formatted_prompt = message
            
            response = requests.post(
                f"{Config.OLLAMA_API_URL}/api/generate",
                json={
                    "model": Config.OLLAMA_MODEL,
                    "prompt": formatted_prompt,
                    "system": "You are LennyBot, a helpful assistant. When responding, reference relevant information from previous messages. Be natural and conversational.",
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 150,
                        "stop": ["\n\n\n", "User:", "Leonardo:"]
                    }
                },
                timeout=timeout
            )
            
            # Log performance
            elapsed = time.time() - start_time
            logging.info(f"Ollama response time: {elapsed:.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                reply = result.get("response", "")
                
                # Validate response
                if not reply or len(reply.strip()) < 2:
                    return "I understand. How else can I help you?"
                
                # Clean response
                reply = reply.split("\n\n")[0]  # Take only first paragraph
                if len(reply) > 300:
                    reply = reply[:300] + "..."
                    
                return reply
            else:
                logging.error(f"Ollama API error: {response.status_code}")
                continue  # Try again if we have attempts left
                
        except Exception as e:
            logging.error(f"Error calling Ollama: {e}")
            if attempt < max_retries - 1:
                continue  # Try again if we have attempts left
    
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