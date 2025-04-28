import os
import csv
import asyncio
import logging
from typing import List, Optional
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
import groq
from datetime import datetime
import pandas as pd

# Load environment variables
load_dotenv()
BOT_TOKEN = "6614402193:AAG30nfyYZpQdCku1rV8IrSjnmjQaazbWIs"
ADMIN_USER_IDS = list(map(int, os.getenv("ADMIN_USER_IDS", "5980915474, 6691432218").split(",")))
GROQ_API_KEY = "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"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize Groq client
groq_client = groq.Groq(api_key=GROQ_API_KEY)

# CSV file path
CSV_FILE = "knowledge_base.csv"

# Ensure CSV file exists with headers
def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'message_link', 'category', 'timestamp'])

# CSV Operations
def add_entry(text: str, message_link: str, category: str = "general") -> bool:
    try:
        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([text, message_link, category, datetime.now().isoformat()])
        return True
    except Exception as e:
        logger.error(f"Error adding entry: {e}")
        return False

def get_entries(category: Optional[str] = None) -> List[dict]:
    entries = []
    try:
        df = pd.read_csv(CSV_FILE)
        if category:
            df = df[df['category'] == category]
        return df.to_dict('records')
    except Exception as e:
        logger.error(f"Error reading entries: {e}")
        return entries

def clear_category(category: str) -> bool:
    try:
        df = pd.read_csv(CSV_FILE)
        df = df[df['category'] != category]
        df.to_csv(CSV_FILE, index=False)
        return True
    except Exception as e:
        logger.error(f"Error clearing category: {e}")
        return False

# Command Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """
Available commands:
/list - List knowledge entries
/add "entry text" "message_link" - Add a new entry (admins only)
/ask <question> - Ask a question using Groq
/download - Download the knowledge base CSV
/clear <category> - Clear entries by category (admins only)
/help - Show this help message
"""
    await update.message.reply_text(help_text)

async def list_entries(update: Update, context: ContextTypes.DEFAULT_TYPE):
    category = None
    if context.args:
        category = context.args[0]
    
    entries = get_entries(category)
    if not entries:
        await update.message.reply_text("No entries found.")
        return

    # Paginate entries
    ENTRIES_PER_PAGE = 5
    pages = [entries[i:i + ENTRIES_PER_PAGE] for i in range(0, len(entries), ENTRIES_PER_PAGE)]
    
    text = "Knowledge Base Entries:\n\n"
    for entry in pages[0]:
        text += f"• {entry['text']}\n  Link: {entry['message_link']}\n\n"
    
    keyboard = []
    if len(pages) > 1:
        keyboard.append([
            InlineKeyboardButton("Next →", callback_data=f"page_1_{category or 'all'}")
        ])
    
    reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
    await update.message.reply_text(text, reply_markup=reply_markup)

async def add_entry_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_USER_IDS:
        await update.message.reply_text("Only admins can add entries.")
        return

    try:
        text, message_link = " ".join(context.args).split('" "')
        text = text.strip('"')
        message_link = message_link.strip('"')
        
        if add_entry(text, message_link):
            await update.message.reply_text("Entry added successfully!")
        else:
            await update.message.reply_text("Failed to add entry.")
    except Exception as e:
        await update.message.reply_text("Usage: /add \"entry text\" \"message_link\"")

async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Please provide a question.")
        return

    question = " ".join(context.args)
    entries = get_entries()
    context_text = "\n".join([f"{e['text']}" for e in entries])
    
    prompt = f"""Context: {context_text}

Question: {question}

Please answer the question based on the context provided. If the answer cannot be found in the context, say so."""

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048,
        )
        answer = completion.choices[0].message.content
        await update.message.reply_text(answer)
    except Exception as e:
        logger.error(f"Error with Groq API: {e}")
        await update.message.reply_text("Sorry, I couldn't process your question.")

async def download(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_document(
            document=open(CSV_FILE, 'rb'),
            filename="knowledge_base.csv"
        )
    except Exception as e:
        await update.message.reply_text(f"Error downloading file: {e}")

async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_USER_IDS:
        await update.message.reply_text("Only admins can clear entries.")
        return

    if not context.args:
        await update.message.reply_text("Please specify a category to clear.")
        return

    category = context.args[0]
    if clear_category(category):
        await update.message.reply_text(f"Cleared all entries in category: {category}")
    else:
        await update.message.reply_text("Failed to clear entries.")

async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    data = query.data.split("_")
    if data[0] == "page":
        page = int(data[1])
        category = data[2] if data[2] != "all" else None
        
        entries = get_entries(category)
        ENTRIES_PER_PAGE = 5
        pages = [entries[i:i + ENTRIES_PER_PAGE] for i in range(0, len(entries), ENTRIES_PER_PAGE)]
        
        if 0 <= page < len(pages):
            text = "Knowledge Base Entries:\n\n"
            for entry in pages[page]:
                text += f"• {entry['text']}\n  Link: {entry['message_link']}\n\n"
            
            keyboard = []
            if page > 0:
                keyboard.append(InlineKeyboardButton("← Prev", callback_data=f"page_{page-1}_{data[2]}"))
            if page < len(pages) - 1:
                keyboard.append(InlineKeyboardButton("Next →", callback_data=f"page_{page+1}_{data[2]}"))
            
            reply_markup = InlineKeyboardMarkup([keyboard]) if keyboard else None
            await query.edit_message_text(text, reply_markup=reply_markup)

def main():
    # Initialize CSV file
    init_csv()
    
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", start))
    application.add_handler(CommandHandler("list", list_entries))
    application.add_handler(CommandHandler("add", add_entry_command))
    application.add_handler(CommandHandler("ask", ask))
    application.add_handler(CommandHandler("download", download))
    application.add_handler(CommandHandler("clear", clear))
    application.add_handler(CallbackQueryHandler(handle_button))

    # Start the bot
    application.run_polling()

if __name__ == '__main__':
    main()
