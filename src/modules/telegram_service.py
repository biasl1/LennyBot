from typing import Tuple, Dict, List, Any, Optional
import logging
import os
import sys
import asyncio
import threading
import time
from telegram import Update
from telegram.ext import Application, ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.request import HTTPXRequest

# Setup paths and imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

from modules.knowledge_store import KnowledgeStore
from modules.meta_context import get_meta_context
from modules.context_scheduler import get_context_scheduler
from modules.user_interaction import store_pin
from modules.decision_agent import decide_action
from modules.action_executor import execute_action

# Global variables
application = None

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# Define the command and message handlers
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    await update.message.reply_text('Hello! I am LennyBot, I can set reminders and answer questions.')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /help is issued."""
    help_text = """
I can help you with:
- Setting reminders (just tell me what to remind you about)
- Answering questions
- General conversation

Just chat with me naturally!
"""
    await update.message.reply_text(help_text)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming messages by queueing them for batch processing."""
    try:
        # Extract message information
        user = update.effective_user
        message_text = update.message.text
        chat_id = update.effective_chat.id
        
        # Log the message but don't process immediately
        logging.info(f"Received message from {user.first_name}: {message_text}")
        
        # Store the message in the meta-context for batch processing
        store_message_for_batch_processing(user, message_text, chat_id)
        
        # Special handling for explicit commands that should be processed immediately
        if message_text.startswith('/'):
            # Handle commands immediately
            await execute_action(update, context, "command")
            
    except Exception as e:
        logging.error(f"Error in handle_message: {e}")

def store_message_for_batch_processing(user, message_text, chat_id):
    """Store the message in meta-context for batch processing with improved metadata."""
    from modules.meta_context import get_meta_context
    meta = get_meta_context()
    
    # Create a standardized pin ID format (be consistent!)
    timestamp = int(time.time())
    pin_id = f"telegram-message_received-{timestamp}"
    
    # Store the message with enhanced metadata
    meta.context_collection.add(
        ids=[pin_id],
        documents=[message_text],  # Store the actual message text
        metadatas=[{
            "event_type": "message_received",
            "user_id": str(user.id),
            "user_name": user.first_name,
            "chat_id": str(chat_id),
            "timestamp": timestamp,  # Use integer timestamp consistently
            "processed": "false",
            "source": "telegram",
            "content_type": "text"
        }]
    )
    
    # Use consistent pin ID format in logging
    logging.info(f"Stored pin: pin-{chat_id}-{timestamp}")

# This function will be called by the application once it's running
async def post_init(application: Application):
    """Initialize the context scheduler after the application has started."""
    # Create and start the context scheduler only if not already running
    context_scheduler = get_context_scheduler(application.bot)
    
    if not context_scheduler.running:
        logging.info("Created new context scheduler")
        # Start the scheduler
        context_scheduler.start()
        logging.info("Context scheduler started successfully!")
        
        # Log this event
        get_meta_context().log_event("system", "scheduler_started", {
            "timestamp": time.time(),
            "type": "context_scheduler"
        })
    else:
        logging.info("Context scheduler already running")

async def process_batched_messages(bot, chat_id, combined_context, messages):
    """Process multiple messages as a batch and send a single response."""
    try:
        # Use a simple typing indicator that won't block too long
        await bot.send_chat_action(chat_id=chat_id, action="typing")
        
        # Determine the most recent intent from these messages
        from modules.decision_agent import get_intent_classifier
        classifier = get_intent_classifier()
        intent, confidence = classifier.classify(combined_context)
        
        logging.info(f"Batch classified as {intent} with confidence {confidence}")
        
        # Generate a single response for all messages
        from modules.response_generator import generate_response
        response = generate_response(combined_context, intent=intent)
        
        # Send a single response
        await bot.send_message(
            chat_id=chat_id,
            text=response,
            parse_mode="HTML"
        )
        
        logging.info(f"Sent batched response to chat {chat_id} for {len(messages)} messages")
        
    except Exception as e:
        logging.error(f"Error in batch processing: {e}")
        # Try a simple fallback message
        try:
            await bot.send_message(
                chat_id=chat_id,
                text="I'm having some trouble processing your messages right now. Let me get back to you shortly."
            )
        except:
            pass

async def context_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display the current conversation context window."""
    chat_id = update.message.chat_id
    
    # Get the time window context (last 10 minutes)
    context_text = get_meta_context().get_unified_context(chat_id, minutes=10)
    
    # Format for display
    response = "🔍 *Your Current Conversation Context:*\n\n"
    response += f"```\n{context_text}\n```"
    
    await update.message.reply_text(response, parse_mode="Markdown")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display the current bot status and metrics."""
    chat_id = update.message.chat_id
    
    # Gather system metrics
    from modules.user_interaction import get_conversation_state, active_conversations
    
    status_text = "🤖 *LennyBot Status Report*\n\n"
    
    # System info
    status_text += "*System:*\n"
    status_text += f"- Uptime: {(time.time() - application.bot_data.get('start_time', time.time()))/60:.1f} minutes\n"
    status_text += f"- Active conversations: {len(active_conversations)}\n"
    
    # User context
    status_text += "\n*Your Context:*\n"
    state = get_conversation_state(chat_id)
    if state:
        status_text += f"- Current intent: {state.get('current_intent', 'none')}\n"
        status_text += f"- Conversation turns: {state.get('turns', 0)}\n"
        ago = time.time() - state.get('last_update', time.time())
        status_text += f"- Last update: {ago:.1f} seconds ago\n"
    else:
        status_text += "- No active conversation\n"
    
    # Intent classifier details 
    status_text += "\n*Intent Classification:*\n"
    from modules.decision_agent import get_classifier_info
    model_info = get_classifier_info()
    status_text += f"- Model: {model_info.get('model_name', 'unknown')}\n"
    status_text += f"- Intents: {', '.join(model_info.get('intents', []))}\n"
    status_text += f"- Confidence threshold: {model_info.get('threshold', 0.5)}\n"
        
    await update.message.reply_text(status_text, parse_mode="Markdown")

# Make sure this function is properly implemented
async def knowledge_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /knowledge command."""
    chat_id = update.effective_chat.id
    user_name = update.effective_user.first_name
    message = update.message.text.replace('/knowledge', '', 1).strip()
    
    # Initialize knowledge store
    from modules.knowledge_store import KnowledgeStore
    knowledge_store = KnowledgeStore()
    
    if message.startswith("add "):
        # Format: /knowledge add [optional:topic] Your knowledge text
        content = message[4:].strip()
        
        # Check if a topic is specified
        parts = content.split(" ", 1)
        if len(parts) > 1 and len(parts[0]) < 20:
            topic = parts[0]
            knowledge_text = parts[1]
            
            knowledge_id = knowledge_store.store_knowledge(
                knowledge_text, 
                topic=topic,
                chat_id=chat_id
            )
            
            if knowledge_id:
                await update.message.reply_text(f"✅ Knowledge added to topic '{topic}'")
            else:
                await update.message.reply_text("❌ Failed to store knowledge")
        else:
            # No topic specified
            knowledge_id = knowledge_store.store_knowledge(
                content,
                chat_id=chat_id
            )
            
            if knowledge_id:
                await update.message.reply_text("✅ Knowledge added")
            else:
                await update.message.reply_text("❌ Failed to store knowledge")
                
    elif message.startswith("search "):
        # Format: /knowledge search Your search query
        query = message[7:].strip()
        
        if not query:
            await update.message.reply_text("Please specify what to search for")
            return
            
        results = knowledge_store.search_knowledge(query, limit=3)
        
        if results:
            response = f"🔍 Results for '{query}':\n\n"
            for i, result in enumerate(results):
                response += f"{i+1}. {result['content']}\n"
                response += f"   Topic: {result['topic']}\n"
                response += f"   Relevance: {result['relevance']:.2f}\n\n"
            
            await update.message.reply_text(response)
        else:
            await update.message.reply_text(f"No knowledge found for '{query}'")
            
    elif message.startswith("topics"):
        # List available topics
        topics = knowledge_store.get_topics()
        
        if topics:
            response = "📚 Available topics:\n\n"
            for topic in topics:
                response += f"• {topic['name']} ({topic['count']} entries)\n"
                
            await update.message.reply_text(response)
        else:
            await update.message.reply_text("No topics found in the knowledge base yet")
            
    else:
        # Help message
        help_text = """
📚 **Knowledge Management Commands**:

• /knowledge add [topic] Your knowledge text
  Adds knowledge to a specific topic

• /knowledge add Your knowledge text
  Adds knowledge with automatic topic detection
  
• /knowledge search Your query
  Searches the knowledge base for information
  
• /knowledge topics
  Lists available knowledge topics
"""
        await update.message.reply_text(help_text, parse_mode="Markdown")

        

# Add to init_handlers function
def init_handlers(application):
    # Existing handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CommandHandler("context", context_command))
    application.add_handler(CommandHandler("status", status_command))
    
    # Add knowledge command
    application.add_handler(CommandHandler("knowledge", knowledge_command))

# Add to setup_telegram_bot function
def setup_telegram_bot():
    """Initialize and configure the Telegram bot."""
    global application
    
    if not Config.TELEGRAM_API_TOKEN:
        logging.error("Telegram API token is not set!")
        return None
    
    logging.info(f"Setting up Telegram bot with token: {Config.TELEGRAM_API_TOKEN[:5]}...{Config.TELEGRAM_API_TOKEN[-5:]}")
    
    # Create the Application instance with post_init callback
    application = (
        ApplicationBuilder()
        .token(Config.TELEGRAM_API_TOKEN)
        .post_init(post_init)  # This will run after the application has started
        .build()
    )
    
    # Add handlers
    init_handlers(application)    
    return application


def start_telegram_bot():
    """Start the Telegram bot."""
    global application
    
    application = setup_telegram_bot()
    
    if application:
        logging.info("Starting Telegram bot...")
        application.run_polling()
    else:
        logging.error("Failed to start Telegram bot.")

# Add this function
def get_bot():
    """Get the Telegram bot instance."""
    global application
    return application.bot if application else None

def create_application(token):
    """Create Telegram application with optimized connection settings."""
    # Fix connection pool issues by increasing pool limits
    request = HTTPXRequest(
        connection_pool_size=8,  # Increase from default
        read_timeout=30,
        write_timeout=30,
        connect_timeout=30,
        pool_timeout=3.0  # Prevent deadlocks
    )
    return Application.builder().token(token).request(request).build()