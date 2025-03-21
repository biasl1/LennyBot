import typing
import sys
sys.modules['typing'] = typing  # Force typing module to be fully loaded
from typing import Tuple, Dict, List, Any, Optional, Union
import logging
import sys
import os
import time
from dotenv import load_dotenv

print("Starting LennyBot...")
start_time = time.time()

# Load environment variables first
load_dotenv(verbose=True)
print(f"[{time.time() - start_time:.2f}s] Environment variables loaded")

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add monkey patch for NumPy 2.0 compatibility
import numpy as np
if not hasattr(np, 'NaN'):
    np.NaN = np.nan  # Add compatibility for libraries using the old attribute
print(f"[{time.time() - start_time:.2f}s] NumPy compatibility configured")

# Create logs directory
os.makedirs("logs", exist_ok=True)

# Configure logging early
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("logs/bot.log"),
        logging.StreamHandler()
    ]
)
print(f"[{time.time() - start_time:.2f}s] Logging configured")

# Log environment variables
telegram_token = os.environ.get("TELEGRAM_API_TOKEN", "")
print(f"Telegram token from env: {telegram_token}")

# Import config first - this is often a source of hangs
print(f"[{time.time() - start_time:.2f}s] Importing config...")
from config import Config
print(f"[{time.time() - start_time:.2f}s] Config imported")
print(f"Telegram token from Config: {Config.TELEGRAM_API_TOKEN}")

# Import meta_context without using signals
print(f"[{time.time() - start_time:.2f}s] Importing meta_context...")
try:
    from modules.meta_context import import_logs_to_history, get_meta_context
    from modules.context_scheduler import get_context_scheduler
    
    # Initialize meta-context
    meta_context = get_meta_context()
    meta_context.log_event("system", "bot_initialized", {
        "timestamp": time.time(),
        "startup_time": time.time() - start_time,
        "version": "1.0.0",
        "environment": os.environ.get("ENVIRONMENT", "development")
    })
    print(f"[{time.time() - start_time:.2f}s] Meta-context system initialized")
    print(f"[{time.time() - start_time:.2f}s] meta_context imported")
except Exception as e:
    print(f"Error initializing meta-context: {e}")

# Try to import logs without signal timeouts
print(f"[{time.time() - start_time:.2f}s] Importing logs...")
try:
    import_logs_to_history()
    print(f"[{time.time() - start_time:.2f}s] Successfully imported previous conversation logs")
except Exception as e:
    print(f"[{time.time() - start_time:.2f}s] Error importing logs: {e}")
    # Continue anyway

# Import the telegram service without signal timeouts
print(f"[{time.time() - start_time:.2f}s] Importing telegram_service...")
try:
    from modules.telegram_service import start_telegram_bot, setup_telegram_bot
    setup_telegram_bot()
    print(f"[{time.time() - start_time:.2f}s] telegram_service imported")
except Exception as e:
    print(f"Error importing telegram_service: {e}")
    sys.exit(1)  # Exit if we can't import the telegram service

# Main function
if __name__ == "__main__":
    # Start the Telegram bot without signal timeouts
    print(f"[{time.time() - start_time:.2f}s] Starting Telegram bot...")
    try:
        start_telegram_bot()
        print(f"[{time.time() - start_time:.2f}s] Telegram bot started")
    except Exception as e:
        print(f"Error starting Telegram bot: {e}")
        sys.exit(1)