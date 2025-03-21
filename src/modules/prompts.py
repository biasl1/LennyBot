import logging
import re
import time
from typing import Dict, Any, Optional, List, Union

class PromptManager:
    """
    Central manager for all prompt engineering in LennyBot.
    This class contains all system prompts, templates, and processing logic.
    """
    
    # ==========================================================================
    # SYSTEM PROMPTS - Personality and behavior instructions for the model
    # ==========================================================================
    SYSTEM_PROMPTS = {
        # General system prompt - default personality
        "general": """YOU ARE LENNYBOT. RESPOND DIRECTLY TO THE USER.

DO:
- Use "I" when referring to yourself
- Keep responses short (25-50 words max)
- Be friendly and helpful
- Respond to exactly what the user asked

DON'T:
- NEVER say "based on the conversation" or "it appears that"
- NEVER analyze what the user is doing
- NEVER say phrases like "the user is asking about"
- NEVER mention "this scenario" or "the given context"

EXAMPLES:
User: "Do you like bananas?"
Good: "Yes, I like bananas! They're delicious and nutritious."
BAD: "Based on the user's query about bananas, it appears they want to know my preferences."

User: "What is your name?"
Good: "I'm LennyBot! Nice to meet you."
BAD: "The user is asking for my identity, which is LennyBot."
""",
        
        # Intent-specific personality adjustments  
        "reminder": """YOU ARE LENNYBOT. YOU SET REMINDERS.

DO:
- Confirm exactly WHAT and WHEN you'll remind the user
- Keep confirmation under 2 sentences
- Be friendly and direct

DON'T:
- NEVER analyze what the user wants
- NEVER start with "Based on" or "It appears"
- NEVER mention "this conversation" or "the user"

EXAMPLE:
User: "Remind me to call mom tomorrow at 5pm"
Good: "I'll remind you to call mom tomorrow at 5pm! Got it."
BAD: "Based on your message, it appears you want a reminder about calling your mother tomorrow at 5pm."
""",
        
        "question": """YOU ARE LENNYBOT. ANSWER QUESTIONS DIRECTLY.

DO:
- Answer briefly (1-3 sentences)
- Respond as if in casual conversation
- Use simple language

DON'T:
- NEVER start with "Based on" or "It appears"
- NEVER analyze the question
- NEVER mention "the user is asking"
- NEVER use phrases like "in this scenario"

EXAMPLE:
User: "What's the capital of France?"
Good: "Paris is the capital of France!"
BAD: "Based on your question about the capital of France, the answer is Paris."
""",
        
        "chat": """YOU ARE LENNYBOT. CHAT NATURALLY.

DO:
- Be conversational and friendly
- Keep responses short (1-3 sentences)
- Use simple language and casual tone
- Respond directly to what was said

DON'T:
- NEVER start with "Based on" or "It appears"
- NEVER analyze the conversation
- NEVER mention "the user" in third person
- NEVER use words like "scenario" or "context"

EXAMPLE:
User: "I'm having a bad day"
Good: "Sorry to hear that! What happened? I'm here if you need to talk."
BAD: "Based on your message, it appears you're experiencing negative emotions today."
"""
    }
    
    # ==========================================================================
    # TEMPLATES - Structured formats for different prompt scenarios
    # ==========================================================================
    TEMPLATES = {
        # Main action execution template
        "action_execution": """CONVERSATION CONTEXT:
{context}

SYSTEM AWARENESS:
- Current intent: {intent}
- Conversation turns: {turns}
- Confidence level: {confidence}

USER MESSAGE: {message}

Based on this context and system state, provide a helpful response. If the conversation has multiple turns, ensure continuity.""",

        # Batch message template for closely timed messages
        "batch_messages": """The following messages are from the same person, sent in quick succession:

{messages}

Respond naturally as if these were part of a single thought.""",

        # Basic context-aware template
        "with_context": """Previous conversation:
{context}

Current message: {message}""",

        # Time-related query template
        "time_query": """The user is asking a time-related question: "{message}"
            
The current time is: {time}
            
Create a friendly, concise response that includes this time information.""",

        # Reminder creation template
        "reminder_creation": """Create a reminder from this message:
"{message}"

Extract:
1. What to remind about
2. When to send the reminder (date and/or time)
3. Any additional details

Format your response as a confirmation of what you'll remind the user about and when.""",

        # Knowledge-enhanced template
        "knowledge_enhanced": """Based on user message: "{message}"
            
RELEVANT KNOWLEDGE:
{knowledge}

Incorporate this knowledge naturally in your response without explicitly mentioning it as a separate source.""",

        # Decision agent template (snowball)
        "snowball_prompt": """You are LennyBot's decision component.

USER MESSAGE: "{message}"

CONVERSATION CONTEXT:
{context}

TASK:
Analyze this message and determine:
1. The user's primary intent (reminder, question, chat)
2. Any specific actions needed
3. A suggested response plan

Output your analysis in valid JSON format with the following fields:
- intent: The primary intent (string)
- confidence: Your confidence in this classification (float between 0-1)
- action_details: Object with action-specific details (if applicable)
- response_plan: A suggested response (string)

For reminder intents, include the following in action_details:
- reminder_text: What to remind about
- time_str: The time expression (e.g., "tomorrow at 3pm")
- parsed_time: Unix timestamp (if you can determine it)""",

        "reminder_creation_confirmation": """I've created a reminder about "{message}" for {time}.
Generate a friendly, natural-sounding confirmation message about this reminder.""",
    }
    
    # ==========================================================================
    # FALLBACK RESPONSES - For when the model fails to generate properly
    # ==========================================================================
    FALLBACKS = {
        "chat": "I understand. Is there anything else you'd like to talk about?",
        "question": "I'm not sure about that. Could you provide more details or rephrase your question?",
        "reminder": "I'd be happy to set a reminder for you. Could you provide more details about what and when?",
        "general": "I'm here to help. What would you like to talk about?",
        "error": "I'm having trouble processing that right now. Could we try again?"
    }
    
    @classmethod
    def get_system_prompt(cls, intent: str = "general") -> str:
        """Get the appropriate system prompt for a given intent/role."""
        return cls.SYSTEM_PROMPTS.get(intent, cls.SYSTEM_PROMPTS["general"])
    
    @classmethod
    def format_prompt(cls, template_name: str, **kwargs) -> str:
        """Format a template with extreme simplicity."""
        if template_name == "with_context":
            context = kwargs.get('context', '')
            message = kwargs.get('message', '')
            
            return f"""PREVIOUS MESSAGES:
{context}

NEW MESSAGE FROM USER:
{message}

RESPOND DIRECTLY TO THE USER'S NEW MESSAGE.
DO NOT START WITH "BASED ON" OR "IT APPEARS".
DO NOT ANALYZE THE CONVERSATION.
JUST REPLY NATURALLY IN A FRIENDLY WAY."""
        
        elif template_name == "reminder_creation_confirmation":
            message = kwargs.get('message', 'something')
            time = kwargs.get('time', 'the specified time')
            
            return f"""User wants reminder: "{message}" at {time}

Confirm this reminder in 1-2 friendly sentences. 
DO NOT analyze their request - just confirm you'll remind them.
DO NOT start with "Based on" or "It appears"."""
        
        # Use default template handling for other cases
        template = cls.TEMPLATES.get(template_name)
        if not template:
            return kwargs.get("message", "")
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            return kwargs.get("message", "")
    
    @classmethod
    def create_action_prompt(cls, message: str, intent: str = "chat", 
                           context: str = "", turns: int = 1,
                           confidence: float = 0.0, **kwargs) -> str:
        """Create a complete prompt for the action executor."""
        # Start with the action execution template
        prompt = cls.format_prompt(
            "action_execution",
            message=message,
            intent=intent,
            context=context,
            turns=turns,
            confidence=confidence
        )
        
        # Add knowledge enhancement if provided
        if "knowledge" in kwargs and kwargs["knowledge"]:
            knowledge_section = f"\n\nRELEVANT KNOWLEDGE:\n{kwargs['knowledge']}"
            prompt += knowledge_section
            
        # Add time information if this is a time query
        if "time" in kwargs and kwargs["time"]:
            time_section = f"\n\nCURRENT TIME: {kwargs['time']}"
            prompt += time_section
            
        return prompt
    
    @classmethod
    def create_batch_prompt(cls, messages: List[str], time_gap: float = 0.0) -> str:
        """Create a simple, direct prompt for batch message processing."""
        if not messages:
            return ""
        
        if len(messages) == 1:
            return messages[0]
        
        # For multiple messages, make it ultra-explicit
        combined = "\n".join([f"Message: {msg}" for msg in messages])
    
        return f"""MESSAGES FROM USER:
{combined}

RESPOND DIRECTLY TO THESE MESSAGES.
IMPORTANT: DO NOT ANALYZE THESE MESSAGES, JUST REPLY TO THEM.
DO NOT START WITH "BASED ON" OR "IT APPEARS".
DO NOT MENTION "THE USER" OR "THESE MESSAGES".
JUST RESPOND AS IF YOU'RE IN A NORMAL CONVERSATION."""
    
    @classmethod
    def create_reminder_prompt(cls, message: str) -> str:
        """Create a reminder-specific prompt."""
        return cls.format_prompt("reminder_creation", message=message)
    
    @classmethod
    def create_decision_prompt(cls, message: str, context: str = "") -> str:
        """Create a prompt for the decision agent."""
        return cls.format_prompt("snowball_prompt", message=message, context=context)
    
    @staticmethod
    def post_process_response(text: str) -> str:
        """Aggressively clean up model responses to make them more human-like."""
        if not text:
            return "I'm here to help! What can I do for you?"
            
        # First, check if the entire response is analytical
        analytical_patterns = [
            r"^based on (the|your|this|our).*",
            r"^it appears that.*",
            r"^in this (scenario|conversation|context).*",
            r"^the user (is asking|wants|needs|mentioned).*"
        ]
        
        for pattern in analytical_patterns:
            if re.match(pattern, text.lower()):
                # Get everything after the analytical prefix
                match = re.search(r"[:,]\s*(.*)", text)
                if match:
                    text = match.group(1)
                else:
                    # If we can't salvage it, replace with a simple response
                    return "I'm here to help you with that!"
        
        # Remove prefixes like "LennyBot:" or "Assistant:"
        prefix_patterns = ["LennyBot:", "Assistant:", "AI:", "Lenny:", "Bot:", "ChatBot:"]
        for prefix in prefix_patterns:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Replace analytical phrases throughout the text
        replacements = [
            (r"based on (the|your|this|our) (conversation|message|context|scenario|query)", ""),
            (r"it appears that", ""),
            (r"i understand that", ""),
            (r"the user is", "you are"),
            (r"the user has", "you have"),
            (r"the user wants", "you want"),
            (r"the user mentioned", "you mentioned"),
            (r"in this (scenario|conversation|context)", ""),
            (r"as mentioned in (your|the) message", ""),
            (r"according to (your|the) message", ""),
            (r"from what i understand", ""),
            (r"from what you've shared", ""),
        ]
        
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace from substitutions
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Fix capitalization after removing phrases
        if text and not text[0].isupper() and len(text) > 1:
            text = text[0].upper() + text[1:]
        
        # Remove user message quotes if the model included them
        text = re.sub(r'You said: ".*?"', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Your message: ".*?"', '', text, flags=re.IGNORECASE)
        
        # Final cleanup of any remaining quotes and whitespace
        text = text.strip('"\'').strip()
        
        return text
    
    @classmethod
    def get_fallback_response(cls, intent: str = "general") -> str:
        """Get an appropriate fallback response for a given intent."""
        return cls.FALLBACKS.get(intent, cls.FALLBACKS["general"])
    
    @classmethod
    def enhance_with_knowledge(cls, prompt: str, knowledge: str) -> str:
        """Add knowledge to a prompt in a standardized way."""
        if not knowledge:
            return prompt
            
        return prompt + f"\n\nRELEVANT KNOWLEDGE:\n{knowledge}\n"
    
    @classmethod
    def log_prompt(cls, prompt: str, response: str, processing_time: float) -> None:
        """Log prompt and response for debugging (shortened versions)."""
        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        response_preview = response[:100] + "..." if len(response) > 100 else response
        
        logging.debug(f"PROMPT: {prompt_preview}")
        logging.debug(f"RESPONSE ({processing_time:.2f}s): {response_preview}")
        
        # More detailed logging at TRACE level if available
        if hasattr(logging, 'TRACE'):
            logging.log(5, f"FULL PROMPT:\n{prompt}")
            logging.log(5, f"FULL RESPONSE:\n{response}")