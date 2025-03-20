import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import os
import random
from sklearn.metrics import accuracy_score 
import numpy as np 


# INTENTS REFERENCE:
# 0: chat - General conversation, greetings, smalltalk
# 1: question - User needs information
# 2: search - User wants to find something
# 3: reminder - User wants to set a reminder
# 4: action - User wants bot to perform a specific task

# More comprehensive training data tailored for your Telegram assistant
# Chat Examples – small talk, greetings, expressions, and casual conversation.
chat_examples = [
    "Hello there!",
    "Hi, how are you doing today?",
    "Good morning, how's everything?",
    "Hey, what's up?",
    "How's it going?",
    "Nice to meet you!",
    "Pleased to see you!",
    "Hey, howdy!",
    "Greetings!",
    "Hi, hope you're doing well.",
    "Hey, good to see you!",
    "Yo, what's going on?",
    "Hello, how have you been?",
    "Hi there, any plans for the day?",
    "Hello, what's new?",
    "Hey, how are you feeling today?",
    "What's happening?",
    "Hi, what's the latest?",
    "How are things?",
    "Hey, everything okay?",
    "Hello! How's your day been so far?",
    "Hi, enjoying your day?",
    "Hey, how do you do?",
    "Hi, just saying hello!",
    "Good afternoon, how are you?",
    "Hi, any fun stories today?",
    "Hey, what's the vibe?",
    "Hello, feeling upbeat today?",
    "Hi, what's on your mind?",
    "Hey, hope you’re having a great day!",
    "Hello, how's life treating you?"
        "Hey",
    "Yo",
    "Hi",
    "Sup?",
    "Morning",
    "Cya",
    "Hola",
    "Heya",
    "Wassup",
    "Yo yo",
    "Hiya",
    "Howdy",
    "Hru?",
    "Sup bro",
    "Hey man",
    "Yo dude",
    "Hi all",
    "Yo all",
    "Greetings",
    "Yo peeps",
    "Mornin'",
    "Aloha",
    "Heya, sup?",
    "Hello",
    "Hiya!",
    "Yo!",
    "Yo, hi",
    "Sup fam",
    "Hey fam",
    "Hi peeps",
    "Yo, what's up?",
    "Hey, what's good?",
    "Hi, what's up?"
]

# Question Examples – fact-based, personal, how-to, and curiosity-driven questions.
question_examples = [
    "What time is it in New York right now?",
    "What's the weather forecast for tomorrow?",
    "How do I make homemade pasta from scratch?",
    "What can you do for me?",
    "Who created you?",
    "When was Bitcoin first introduced?",
    "Why is the sky blue on a clear day?",
    "Can you help me figure out my schedule?",
    "How does this bot work exactly?",
    "What are your capabilities and limitations?",
    "Do you have access to real-time information?",
    "How tall is the Eiffel Tower in meters?",
    "What's the capital of France?",
    "How many planets are there in our solar system?",
    "What's the meaning of life according to philosophy?",
    "What is the history behind the Great Wall of China?",
    "How do quantum computers work?",
    "Can you explain string theory in simple terms?",
    "What are the symptoms of the flu?",
    "When is the next full moon?",
    "How do I reset my password?",
    "What does the term 'machine learning' mean?",
    "How can I optimize my computer performance?",
    "What is the best way to learn Python?",
    "Why do leaves change color in the fall?",
    "Can you tell me the current stock price of Tesla?",
    "What is the process for applying for a visa?",
    "How do I bake a chocolate cake?",
    "What are the side effects of caffeine?",
    "Can you help me understand blockchain technology?",
    "Why is exercise important for health?"
        "Time?",
    "Weather?",
    "How work?",
    "Why blue?",
    "Who u?",
    "Cap France?",
    "Bitcoin?",
    "How pasta?",
    "Food?",
    "Where go?",
    "What news?",
    "How tech?",
    "Cost?",
    "Money?",
    "TV?",
    "Game?",
    "Who win?",
    "When?",
    "How?",
    "Where?",
    "Which?",
    "Why?",
    "What app?",
    "Score?",
    "Tweet?",
    "How many?",
    "Dose?",
    "Pic?",
    "Link?",
    "URL?",
    "Mins?",
    "Sec?",
    "Time now?",
    "Date?",
    "Temp?",
    "Air?",
    "Run?",
    "Build?",
    "Drive?",
    "Fuel?",
    "News?"
]

# Search Examples – queries targeting local results, recipes, tech info, and entertainment.
search_examples = [
    "Search for the best Italian restaurants near me",
    "Find top-rated pizza recipes with homemade dough",
    "Look up the weather in New York City for this weekend",
    "Search for the latest international news updates",
    "Find detailed information about quantum computing breakthroughs",
    "Look for comprehensive JavaScript tutorials for beginners",
    "Search for nearby coffee shops with free Wi-Fi",
    "Find the definition and meaning of serendipity",
    "Look up cheap flights to London for next month",
    "Search for recommended Python programming books for advanced developers",
    "Find high-resolution images of cats and kittens",
    "Look up movie showtimes in my area",
    "Search for detailed electric car reviews and comparisons",
    "Find hiking trails near my current location with user ratings",
    "Look up the full lyrics to Bohemian Rhapsody by Queen",
    "Search for vegan dessert recipes",
    "Find top-rated science fiction novels from the past decade",
    "Look up the schedule for upcoming tech conferences",
    "Search for DIY home improvement ideas on a budget",
    "Find travel guides for a road trip across Europe",
    "Search for meditation and mindfulness apps",
    "Find healthy lunch recipes under 500 calories",
    "Look up local farmers' markets near my area",
    "Search for latest smartphone reviews and benchmarks",
    "Find comprehensive tutorials on data science with Python",
    "Look up scenic spots for photography in the countryside",
    "Search for beginner’s guides to investing in stocks",
    "Find podcasts about history and culture",
    "Look up recipes for gluten-free baking",
    "Search for local events happening this weekend",
    "Find tutorials on setting up a home network",
    "Look up best practices for remote working",
    "Search for inspirational quotes about success and perseverance"
        "coffee near me",
    "pizza recipe",
    "news update",
    "movie times",
    "tech news",
    "dog pics",
    "convert 30EUR",
    "stocks",
    "map",
    "jobs",
    "cats",
    "memes",
    "recipes",
    "bacon",
    "weather",
    "bar near me",
    "gas station",
    "local events",
    "concert",
    "reviews",
    "tips",
    "guide",
    "DIY",
    "tutorial",
    "tutorials",
    "discounts",
    "offers",
    "sales",
    "deals",
    "video",
    "how to",
    "menu",
    "restaurants",
    "trending",
    "sports",
    "scores",
    "updates",
    "coupons",
    "calendar",
    "fitness",
    "health",
    "bitcoin",
    "crypto",
    "apps",
    "games",
    "music",
    "podcast",
    "news"
]

# Reminder Examples – setting reminders with various details and contextual phrasing.
reminder_examples = [
    "Remind me to buy milk tomorrow morning",
    "Set a reminder for my meeting at 3 PM this afternoon",
    "Don't let me forget to call my mom later today",
    "Remind me to send that important email in an hour",
    "Set an alarm for 7 AM tomorrow for my workout",
    "Remind me to take my medicine at 9 PM tonight",
    "Can you remind me to be kind to myself later?",
    "Set a reminder about my dentist appointment next Tuesday",
    "Remind me to water the plants every day at 8 AM",
    "Set a reminder for my anniversary next week",
    "Don't forget to remind me about the party this Friday",
    "Remind me to check the oven in 20 minutes",
    "Set a reminder for the game tonight at 7 PM",
    "Remind me to lock the door when I leave the house",
    "Please remind me to finish my homework before dinner",
    "Set an alarm for my morning meditation at 6 AM",
    "Remind me to review the project proposal this afternoon",
    "Can you remind me to call the electrician tomorrow?",
    "Set a reminder to schedule a follow-up appointment next week",
    "Remind me to update my calendar with next month’s events",
    "Please remind me to pick up my dry cleaning after work",
    "Set an alarm for my early flight tomorrow morning",
    "Remind me to send a birthday message to Sarah",
    "Can you remind me to check the mail in the evening?",
    "Set a reminder to pay the utility bills on the 5th of each month",
    "Remind me to confirm my reservation at the restaurant tonight",
    "Please remind me to take a short walk during my break",
    "Set a reminder to review my finances at the end of the week",
    "Remind me to research gift ideas for my friend’s birthday",
    "Can you remind me to update my password every three months?",
    "Set an alarm for my study session at 8 PM tonight",
    "Remind me to prepare my presentation slides by tomorrow morning",
    "Please remind me to check in with the team later today",
    "Set a reminder for my virtual meeting at 10 AM",
    "Remind me to review the new policy documents this evening",
    "Can you remind me to get some fresh air after lunch?",
    "Set a reminder to organize my workspace at the end of the day"
        "coffee in 5 min",
    "call mom 2pm",
    "meds 9pm",
    "alarm 7am",
    "mtg 3pm",
    "oven in 10",
    "gym at 6",
    "mail later",
    "pick up 5",
    "doc 4",
    "car svc 3",
    "lunch 12",
    "dinner 7",
    "water plants",
    "pay bill",
    "meet Tom",
    "doc 2pm",
    "class 8",
    "read book",
    "reply email",
    "check bank",
    "order food",
    "buy gift",
    "clean room",
    "run errand",
    "fix bike",
    "grocery run",
    "water garden",
    "start laundry",
    "take meds",
    "timer 15",
    "vacuum",
    "call friend",
    "take break",
    "charge phone",
    "pick up kids",
    "call taxi",
    "doc appt",
    "sched mtg",
    "upd cal",
    "pay rent",
    "meds now",
    "nap in 10",
    "read news"
]

# Action Examples – commands to perform actions like messaging, creating tasks, calculations, etc.
action_examples = [
    "Send a message to John saying, 'I'll be there in 10 minutes.'",
    "Create a new task to call the plumber about the leak in the kitchen",
    "Add eggs, milk, and bread to my shopping list",
    "Start a timer for 10 minutes for my break",
    "Schedule a meeting with the team for Monday morning at 9 AM",
    "Create a note outlining the project ideas discussed today",
    "Translate the following sentence to Spanish: 'How are you doing today?'",
    "Calculate 15% of 67.50 and show me the result",
    "Convert 30 euros to US dollars using the current exchange rate",
    "Take a screenshot of my current desktop",
    "Share my current location with Lisa",
    "Play some upbeat music from my favorites playlist",
    "Turn on the living room lights",
    "Send an email to support with my issue details",
    "Start my workout routine by playing a training video",
    "Open the calendar app and show me my schedule for today",
    "Set a reminder to call the bank about my account",
    "Download the latest report from the project folder",
    "Launch the calculator app",
    "Open the weather app and display the forecast for today",
    "Update my status on social media to 'Busy at work'",
    "Search for the nearest gas station and navigate there",
    "Set the thermostat to 72 degrees Fahrenheit",
    "Mute all notifications for the next 30 minutes",
    "Create a shopping list for the weekend groceries",
    "Find and open the document titled 'Project Plan'",
    "Enable Bluetooth on my device",
    "Switch the display mode to dark theme",
    "Read me the latest news headlines",
    "Play a podcast about technology trends",
    "Show me my recent emails from last week",
    "Open the map and find the best route home",
    "Set a timer for a 15-minute meditation session",
    "Schedule an appointment with Dr. Smith for next Monday",
    "Call my best friend and ask how they're doing",
    "Remind me to update my software later today",
    "Locate the nearest coffee shop and display directions",
    "Create an event for my birthday party next month",
    "Order a large pepperoni pizza from my favorite restaurant",
    "Open the music app and shuffle my workout playlist",
    "Show me the conversion for 100 miles to kilometers"
        "msg John",
    "call plmr",
    "add eggs",
    "timer 10min",
    "calc 15% of 67.5",
    "screenshot",
    "open app",
    "play song",
    "set alarm",
    "send email",
    "download file",
    "start vid",
    "turn on lights",
    "mute mic",
    "next song",
    "prev song",
    "pause",
    "resume",
    "stop",
    "restart",
    "book ride",
    "locate store",
    "pay bill",
    "order food",
    "open map",
    "take pic",
    "snap shot",
    "check mail",
    "reply msg",
    "log out",
    "login",
    "start game",
    "open browser",
    "close app",
    "switch tab",
    "share link",
    "upload pic",
    "print doc",
    "scan code",
    "open file",
    "play vid",
    "show weather",
    "find deal",
    "flip coin",
    "calc tip",
    "set timer 5",
    "start workout",
    "stop timer",
    "next vid",
    "prev vid"
]


# Combine all examples
texts = chat_examples + question_examples + search_examples + reminder_examples + action_examples
intents = ([0] * len(chat_examples) + 
           [1] * len(question_examples) + 
           [2] * len(search_examples) +
           [3] * len(reminder_examples) +
           [4] * len(action_examples))

# Create dataset
data = {'text': texts, 'intent': intents}

# Shuffle the data to mix the categories
combined = list(zip(texts, intents))
random.shuffle(combined)
texts, intents = zip(*combined)

data = {
    'text': texts,
    'intent': intents
}

# Convert to dataset
df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

# Split into training and validation sets
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=5
)
def tokenize_function(examples):
    result = tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True,
        max_length=128
    )
    
    # Copy the intent column to labels (THIS IS THE IMPORTANT FIX)
    result["labels"] = examples["intent"]
    return result

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Make sure we keep only the columns the model needs
columns_to_keep = ['input_ids', 'attention_mask', 'labels']
tokenized_train.set_format(type='torch', columns=columns_to_keep)
tokenized_test.set_format(type='torch', columns=columns_to_keep)


# Set proper output directory
model_path = "/Users/test/Documents/development/LennyBot/lennybot/src/models/intent_classifier"
results_path = "/Users/test/Documents/development/LennyBot/lennybot/src/models/training_results"


# Add this function before creating the Trainer
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Training arguments - improved for better convergence
training_args = TrainingArguments(
    output_dir=results_path,
    per_device_train_batch_size=8,     # Smaller batch size
    per_device_eval_batch_size=16,
    num_train_epochs=20,               # More epochs
    weight_decay=0.01,
    learning_rate=2e-5,                # Slightly lower learning rate
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    # Add warmup steps
    warmup_ratio=0.1,
)

# Define Trainer with evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,  # Add this line
)

# Train the model
print("Starting model training...")
trainer.train()

# Evaluate the model
print("Evaluating model...")
results = trainer.evaluate()
print(f"Evaluation results: {results}")

# Save the model
os.makedirs(model_path, exist_ok=True)
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print(f"Model and tokenizer saved to {model_path}")