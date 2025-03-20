import logging
import os
import sys
import asyncio
import threading
from telegram import Update
from telegram.ext import Application, ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from modules.reminder_scheduler import ReminderScheduler
import time
from modules.conversation_context import get_time_window_context, get_recent_context
from modules.knowledge_store import KnowledgeStore


# Setup paths and imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

from modules.user_interaction import store_pin
from modules.decision_agent import decide_action
from modules.action_executor import execute_action

active_conversations = {}  # Track active conversations by chat_id


# Global variables
application = None
reminder_scheduler = None

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

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process incoming message and respond using our decision pipeline."""
    user_message = update.message.text
    user_name = update.effective_user.first_name
    chat_id = update.message.chat_id

    logger.info(f"Received message from {user_name}: {user_message}")

    # Store the pin
    pin = store_pin(chat_id, user_message)

    # Decide on the action
    action = decide_action(pin)

    # Execute the action (send the response)
    await execute_action(update, context, action)

# This function will be called by the application once it's running
async def post_init(application: Application):
    """Initialize the reminder scheduler after the application has started."""
    global reminder_scheduler
    
    # Create reminder scheduler
    reminder_scheduler = ReminderScheduler(application.bot)
    logging.info("Created new reminder scheduler")
    
    # Start the scheduler
    await reminder_scheduler.start()
    logging.info("Reminder scheduler started successfully!")
async def context_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display the current conversation context window."""
    chat_id = update.message.chat_id
    
    # Get the time window context (last 10 minutes)
    context_text = get_time_window_context(chat_id, minutes=10)
    
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
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CommandHandler("context", context_command))
    application.add_handler(CommandHandler("status", status_command))

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


        