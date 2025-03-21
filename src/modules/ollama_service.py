import requests
import logging
import time
import re
from config import Config
from modules.prompts import PromptManager

def process_message(message, system_role="general", conversation_history=None, verbose=False):
    """
    Process a message through Ollama and return the response.
    
    Args:
        message (str): The message to process
        system_role (str): The role/intent to use for selecting the system prompt
        conversation_history (str): Optional conversation history
        verbose (bool): Whether to log verbose information
        
    Returns:
        str: The processed response
    """
    start_time = time.time()
    
    try:
        # Get the appropriate system prompt using the PromptManager
        system_prompt = PromptManager.get_system_prompt(system_role)
        
        # Format the complete prompt with conversation history if provided
        if conversation_history:
            full_prompt = PromptManager.format_prompt(
                "with_context",
                context=conversation_history,
                message=message
            )
        else:
            full_prompt = message
            
        # Log what we're sending to Ollama
        logging.info(f"Sending prompt to Ollama: {full_prompt[:100]}...")
        
        # Send to Ollama
        response_data = send_to_ollama(full_prompt, system_prompt)
        
        # Extract and process the response text
        if isinstance(response_data, dict) and "response" in response_data:
            response_text = response_data["response"].strip()
            # Log the actual response we got
            logging.info(f"Raw Ollama response text: {response_text[:100]}...")
            
            # Post-process using the PromptManager
            response_text = PromptManager.post_process_response(response_text)
            
        else:
            logging.error(f"Unexpected response format from Ollama: {response_data}")
            response_text = PromptManager.get_fallback_response("error")
        
        # Measure and log response time
        elapsed = time.time() - start_time
        logging.info(f"Ollama response time: {elapsed:.2f}s")
        
        # Log prompt and response for debugging if verbose
        if verbose:
            PromptManager.log_prompt(full_prompt, response_text, elapsed)
        
        return response_text
    except Exception as e:
        logging.error(f"Error processing message with Ollama: {e}")
        return PromptManager.get_fallback_response("error")

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
            return {"response": "I'm having trouble connecting to my thinking module."}
            
    except Exception as e:
        logging.error(f"Error in send_to_ollama: {e}")
        return {"response": "I encountered an error while processing your request."}