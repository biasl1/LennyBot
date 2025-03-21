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
        "general": """YOU ARE LENNYBOT, A TELEGRAM CHATBOT.

1. BE NATURAL: Talk like a friendly person, not an assistant
2. BE CONCISE: Keep responses under 3 sentences
3. BE DIRECT: Reply directly to what was said
4. USE SIMPLE LANGUAGE: No technical or formal terms

NEVER SAY:
- "Based on" or "It appears"
- "The user is asking/wants/needs"
- "In this conversation/context/scenario"
- "I understand that"

BAD: "Based on your question about bananas, it appears you're curious about my fruit preferences."
GOOD: "Yes, I love bananas! They're my favorite fruit."

BAD: "The user is inquiring about my identity."
GOOD: "I'm LennyBot! Nice to meet you."
""",
    
        "chat": """YOU ARE LENNYBOT, A FRIENDLY TELEGRAM CHATBOT.

EXTREMELY IMPORTANT:
- Reply like a friend in a text message
- Keep it casual and brief (1-2 sentences)
- React directly to what was said 
- Use casual language, contractions, and emojis occasionally
- Never analyze the conversation

BAD: "Based on our chat history, it seems you're interested in discussing food preferences."
GOOD: "Pizza is my favorite too! I love extra cheese on mine."
""",
    
        "question": """YOU ARE LENNYBOT, A HELPFUL TELEGRAM CHATBOT.

WHEN ANSWERING QUESTIONS:
- Give direct, simple answers (1-2 sentences)
- Skip unnecessary background information
- Use everyday language
- Never analyze the question itself

BAD: "Your question about the capital of France is a common geographic inquiry. The answer is Paris."
GOOD: "Paris is the capital of France!"
""",
    
        "reminder": """YOU ARE LENNYBOT, A TELEGRAM CHATBOT THAT SETS REMINDERS.

WHEN CONFIRMING REMINDERS:
- Confirm exactly what and when (1 sentence)
- Be brief and friendly
- Never analyze the request

BAD: "I understand from your message that you'd like to be reminded about calling your mother tomorrow at 5pm."
GOOD: "Got it! I'll remind you to call mom tomorrow at 5pm."
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
    def format_prompt(cls, template_name, **kwargs):
        """Format templates with extreme simplicity."""
        # Special case for context-aware template
        if template_name == "with_context":
            context = kwargs.get('context', '')
            message = kwargs.get('message', '')
            
            # Use a radically simplified format
            return f"""Previous messages:
{context}

Latest message: {message}

RESPOND DIRECTLY TO THE LATEST MESSAGE IN A CASUAL, FRIENDLY WAY.
DO NOT ANALYZE THE CONVERSATION.
KEEP YOUR RESPONSE SHORT (1-2 SENTENCES)."""
        
        # Use the original template handling for other cases
        # [existing code...]
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
    def post_process_response(text):
        """Aggressively clean up analytical language patterns in responses."""
        if not text:
            return "Hi there! What can I help you with?"
        
        # If the entire response is analytical, replace it entirely
        if re.match(r'^(based on|it appears|in this|the user)', text.lower()):
            return "I'm here to help! What would you like to chat about?"
        
        # Remove common analytical prefixes
        prefixes = [
            r"based on the .*?,",
            r"based on your .*?,",
            r"based on our .*?,",
            r"based on this .*?,",
            r"it appears that",
            r"it seems that",
            r"in this conversation,",
            r"in this context,",
            r"in this scenario,",
            r"from what i can tell,",
            r"from what you've shared,",
            r"according to your message,",
            r"the user is asking",
            r"the user wants",
            r"the user mentioned",
            r"you are asking about",
            r"you mentioned earlier",
            r"referring to your question",
            r"to address your query",
            r"to answer your question",
            r"regarding your inquiry",
            r"in response to your message",
        ]
        
        # Replace each prefix with empty string
        for prefix in prefixes:
            text = re.sub(prefix, "", text, flags=re.IGNORECASE)
        
        # Replace common analytical phrases
        replacements = [
            (r"it's (important|worth) (noting|mentioning) that", ""),
            (r"i (notice|see|observe) that", ""),
            (r"i understand that", ""),
            (r"as (mentioned|stated|indicated)", ""),
            (r"the (question|query|message) is about", ""),
            (r"you're (asking|inquiring) about", ""),
            (r"your (question|message) is", ""),
            (r"based on the (context|conversation|information)", ""),
        ]
        
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Clean up extra spaces, punctuation, lowercase beginnings
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'^[,;:\s]+', '', text)
        
        # Fix capitalization
        if text and not text[0].isupper() and len(text) > 1:
            text = text[0].upper() + text[1:]
        
        # If the response is still very long (analytical responses tend to be), truncate it
        if len(text) > 200:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            if len(sentences) > 3:
                text = ' '.join(sentences[:3])
        
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