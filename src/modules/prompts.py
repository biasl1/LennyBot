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
        "general": """You are LennyBot, a helpful and friendly Telegram assistant.
Respond directly to users in a natural, conversational way.
Don't use prefixes like 'LennyBot:' or formatting markers.
Keep responses brief, helpful and human-like.
Be conversational but concise.""",
            
        # Intent-specific personality adjustments
        "reminder": """You are LennyBot, a helpful assistant specialized in managing reminders.
Focus on extracting time information and reminder content accurately.
Respond naturally and confirm the details you've understood.
When setting reminders, clearly repeat back the time and task to confirm.
If time is ambiguous, ask for clarification.""",
            
        "question": """You are LennyBot, a knowledgeable assistant.
Provide accurate, clear answers to questions.
When unsure, admit limitations rather than making up information.
Keep responses concise and informative.
If given knowledge to reference, incorporate it naturally without mentioning the source.""",
            
        "chat": """You are LennyBot, a friendly conversational assistant.
Engage in natural dialogue, showing personality but remaining concise.
Respond directly to the user's statements or questions.
Keep the conversation flowing naturally and show empathy where appropriate.
Remember details the user has shared previously.""",
            
        # Special purpose prompts
        "time_aware": """You are LennyBot, a helpful assistant with time awareness.
Include the provided time information in your response when relevant.
Be precise and helpful when discussing time-related information.
Format times in a user-friendly way.""",
            
        "knowledge_enhanced": """You are LennyBot, a knowledgeable assistant.
Use the provided knowledge to inform your response.
Incorporate this information naturally without explicitly mentioning that you were given additional context.
If the knowledge conflicts with the user's statement, gently correct them while being respectful.""",
            
        "decision": """You are LennyBot's decision component.
Analyze the user's message carefully to determine:
1. The underlying intent (reminder, question, chat)
2. Any specific actions that need to be taken
3. Relevant details needed for those actions
Be precise in your classification but don't explain your reasoning."""
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
        """Format a template with the provided variables."""
        template = cls.TEMPLATES.get(template_name)
        if not template:
            logging.warning(f"Template '{template_name}' not found, using raw message")
            return kwargs.get("message", "")
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logging.error(f"Missing key in template '{template_name}': {e}")
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
        """Create a prompt for batch message processing."""
        combined_text = "\n".join(messages)
        
        if len(messages) > 1:
            return cls.format_prompt("batch_messages", messages=combined_text)
        else:
            return combined_text
    
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
        """Clean up model responses to make them more human-like."""
        if not text:
            return ""
            
        # Remove "LennyBot:" prefix if present
        if text.startswith("LennyBot:"):
            text = text[len("LennyBot:"):].strip()
        
        # Remove any other response pattern like "Assistant:" or "AI:"
        patterns = ["Assistant:", "AI:", "Lenny:", "Bot:", "ChatBot:"]
        for pattern in patterns:
            if text.startswith(pattern):
                text = text[len(pattern):].strip()
        
        # Remove conversation format if model generated both sides
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Skip lines that look like user messages
            if any(line.strip().startswith(prefix) for prefix in ["User:", "Human:", "You:"]):
                continue
            # Remove assistant prefixes within lines
            for prefix in patterns:
                if prefix in line:
                    line = line.replace(prefix, "")
            cleaned_lines.append(line)
        
        # Rejoin and clean up extra whitespace
        text = '\n'.join(cleaned_lines)
        text = re.sub(r'\n{3,}', '\n\n', text)  # Replace excessive newlines
        
        return text.strip()
    
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