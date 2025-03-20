# LennyBot

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Telegram Bot API](https://img.shields.io/badge/Telegram%20Bot%20API-latest-blue.svg)](https://core.telegram.org/bots/api)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4.0+-green.svg)](https://github.com/chroma-core/chroma)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An intelligent Telegram assistant with conversation memory, intent classification, and proactive interactions**

</div>

## 🌟 Features

- **🧠 Contextual Memory**: Maintains multi-turn conversations using ChromaDB
- **🔍 AI Intent Classification**: Fine-tuned DistilBERT model understands user intentions
- **⏰ Smart Reminders**: Set and receive reminders with natural language
- **🔄 Time Window Architecture**: Maintains coherent conversation state across time
- **🚀 Proactive Responses**: Follows up on incomplete conversations automatically
- **🛠️ Local LLM Integration**: Powered by Ollama for privacy-focused AI responses
- **📊 Modular Architecture**: Easily extend with new capabilities

## 📋 Installation

### Prerequisites

- Python 3.11 or higher
- [Ollama](https://ollama.ai) installed locally
- Telegram Bot Token (obtain from [BotFather](https://t.me/botfather))

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/lennybot.git
cd lennybot

# Create and activate virtual environment
python -m venv .venv_py311
source .venv_py311/bin/activate  # Windows: .venv_py311\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
# Edit .env to add your Telegram token and other settings

# Run the bot
cd src
python main.py
```

## 💻 Architecture

LennyBot uses a sophisticated time-based context architecture:

### Core Components

- **Decision Agent**: Classifies intents and determines appropriate actions
- **Conversation Context**: Manages time-window based memory system
- **Action Executor**: Handles user requests based on classified intents
- **Reminder Scheduler**: Background service for timed reminders
- **Intent Classifier**: DistilBERT model fine-tuned for conversation understanding

### Data Flow

1. **User Message** → Telegram API
2. **Intent Classification** → Identifies user's intention (chat, question, search, reminder, action)
3. **Context Retrieval** → Gets relevant conversation history from ChromaDB
4. **LLM Processing** → Generates appropriate response using context
5. **Response Delivery** → Sends message to user
6. **Memory Storage** → Archives conversation in vector database
7. **Proactive Monitoring** → Checks for incomplete conversations and follows up

## 🚀 Usage

### Commands

- `/start` - Initialize the bot
- `/help` - Display available commands and capabilities
- `/context` - Show current conversation context window

### Example Interactions

```
User: Hello!
LennyBot: Hi there! How can I help you today?

User: Can you remind me to call mom?
LennyBot: Sure! When would you like to be reminded to call mom?

User: In 10 minutes
LennyBot: Got it! I'll remind you to call mom in 10 minutes.

[10 minutes later]
LennyBot: ⏰ Reminder: Call mom
```

## 🔧 Configuration

Edit the `.env` file to customize:

- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token
- `OLLAMA_API_URL`: URL for your Ollama instance
- `OLLAMA_MODEL`: LLM model to use (phi, mistral, llama2, etc.)
- `CHROMADB_PATH`: Storage location for conversation history

## 📚 Development

### Intent Classification

LennyBot uses a fine-tuned DistilBERT model for intent classification:

- **Chat**: General conversation, greetings, smalltalk
- **Question**: User needs information
- **Search**: User wants to find something
- **Reminder**: User wants to set a reminder
- **Action**: User wants bot to perform a specific task

To fine-tune the model with new examples:

```bash
python scripts/finetune_intent_model.py
```

### Time Window Architecture

The bot maintains conversation state by:

1. Storing all interactions in ChromaDB
2. Establishing a time window (default: 10 minutes)
3. Using historical context to inform current responses
4. Detecting incomplete conversations and following up proactively

## 🧪 Troubleshooting

- **Bot not responding**: Verify Telegram token and ensure Ollama is running
- **Reminders not working**: Check system time and reminder scheduler logs
- **Memory issues**: Ensure ChromaDB has proper permissions at the specified path
- **High latency**: Consider using a lighter LLM model in your Ollama configuration

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a pull request

## 📞 Contact

For questions and support, please open an issue in the GitHub repository.

---

<div align="center">
<p>Made with ❤️ by Leonardo</p>
</div>