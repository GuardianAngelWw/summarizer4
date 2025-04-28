import os
import csv
import asyncio
import logging
import re
import tempfile
import json
from typing import List, Dict, Optional, Tuple, Any, Set
from functools import wraps
from collections import deque
from datetime import datetime
import pytz
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton, ChatMember, Chat
from telegram.ext import (
    Application, 
    CommandHandler, 
    MessageHandler, 
    CallbackQueryHandler,
    ContextTypes, 
    filters
)
from telegram.constants import ParseMode, ChatType
import aiohttp  # NEW: For async HTTP requests to Groq API
from flask import Flask, jsonify
import logging.handlers
import waitress
import threading

# ------------------ NEW: GROQ API CONSTANTS ------------------
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"  # Or your preferred model

# ------------------ BOT STATUS & LOGGING ------------------
STARTUP_MESSAGE = """
🤖 Bot Status Update 🤖
Status: Online ✅
Time: {}
Version: {}
Environment: GitHub Actions
"""

SHUTDOWN_MESSAGE = """
🤖 Bot Status Update 🤖
Status: Offline ⛔
Time: {}
Reason: {}
"""

HEALTH_CHECK_MESSAGE = """
🏥 Health Check Report 🏥
Status: {}
Time: {}
Memory Usage: {:.2f}MB
Uptime: {}
Active Users: {}
"""

class BotStatusMonitor:
    def __init__(self, bot_token: str, admin_ids: list[int]):
        self.bot_token = bot_token
        self.admin_ids = admin_ids
        self.start_time = datetime.now(pytz.UTC)
        self.active_users = set()
        self.version = "2025.04.27"  # Update this with your version

    async def send_to_admins(self, message: str):
        """Send a message to all admin users."""
        app = Application.builder().token(self.bot_token).build()
        for admin_id in self.admin_ids:
            try:
                await app.bot.send_message(
                    chat_id=admin_id,
                    text=message,
                    parse_mode='HTML'
                )
            except Exception as e:
                logging.error(f"Failed to send status to admin {admin_id}: {e}")
        await app.shutdown()

    def get_memory_usage(self):
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def get_uptime(self):
        """Get bot uptime in human readable format."""
        delta = datetime.now(pytz.UTC) - self.start_time
        days = delta.days
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{days}d {hours}h {minutes}m {seconds}s"

    async def send_startup_notification(self):
        """Send startup notification to admins."""
        message = STARTUP_MESSAGE.format(
            datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S"),
            self.version
        )
        await self.send_to_admins(message)

    async def send_shutdown_notification(self, reason: str = "Planned Shutdown"):
        """Send shutdown notification to admins."""
        message = SHUTDOWN_MESSAGE.format(
            datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S"),
            reason
        )
        await self.send_to_admins(message)

    async def send_health_check(self):
        """Send health check status to admins."""
        status = "Healthy ✅"
        try:
            # Add your health checks here
            # For example, check database connection, API status, etc.
            pass
        except Exception as e:
            status = f"Warning ⚠️\nError: {str(e)}"

        message = HEALTH_CHECK_MESSAGE.format(
            status,
            datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S"),
            self.get_memory_usage(),
            self.get_uptime(),
            len(self.active_users)
        )
        await self.send_to_admins(message)

# Load environment variables first
load_dotenv()

# Add a deque to store the last 10 log messages
last_logs = deque(maxlen=10)

# Setup log file
# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)
log_file = os.path.join('logs', 'bot.log')

# Custom log handler to capture logs in memory
class MemoryLogHandler(logging.Handler):
    def emit(self, record):
        try:
            # Format the log message
            msg = self.format(record)
            # Add timestamp in UTC
            timestamp = datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')
            formatted_msg = f"{timestamp} - {msg}"
            # Add to deque
            last_logs.append(formatted_msg)
        except Exception:
            self.handleError(record)

# Initialize logger first
logger = logging.getLogger(__name__)

# Configuration
BOT_TOKEN = "6614402193:AAHGTmV-ZXSKbhd9_UGvux2AVrVbXyiUbeE" # Bot token should be provided via environment variable

# Modify the logging setup (around line 55)
if not logging.getLogger().handlers:
    log_format = "%(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10485760,  # 10MB
                backupCount=5
            ),
            MemoryLogHandler()  # Add the memory handler
        ]
    )

# Update the env variables (around line 37)
ADMIN_USER_IDS = os.getenv("ADMIN_USER_IDS", "6691432218, 5980915474").strip()
ADMIN_USERS = [int(uid.strip()) for uid in ADMIN_USER_IDS.split(",")] if ADMIN_USER_IDS else []
CURRENT_DATE = "2025-04-27 09:19:30"  # Updated current UTC time
CURRENT_USER = "GuardianAngelWw"      # Updated current user
ENTRIES_FILE = "entries.csv"
CATEGORIES_FILE = "categories.json"
CSV_HEADERS = ["text", "link", "category", "group_id"]  # Added category and group_id fields

# Flask app initialization
app = Flask(__name__)

# Modify the startup logging to be more secure (around line 72)
logger.info(f"Bot starting at {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
logger.info(f"Running on device: cloud (Groq API)")
logger.info(f"Model: {GROQ_MODEL} (via Groq API)")
logger.info("Bot initialization successful")  # Instead of logging the token

# Flask routes for health monitoring
@app.route('/health')
def health_check():
    """Health check endpoint for container monitoring"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    }), 200

@app.route('/')
def root():
    """Root endpoint with basic information"""
    return jsonify({
        'service': 'Summarizer Bot',
        'status': 'running',
        'documentation': '/health for health check endpoint'
    }), 200

# Pagination configuration
ENTRIES_PER_PAGE = 5

# Create entries.csv and categories.json if they don't exist
if not os.path.exists(ENTRIES_FILE):
    with open(ENTRIES_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)

if not os.path.exists(CATEGORIES_FILE):
    with open(CATEGORIES_FILE, "w", encoding="utf-8") as f:
        json.dump({"categories": ["General", "Documentation", "Tutorials", "References"]}, f)

# Helper Functions
async def is_admin(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int) -> bool:
    """Check if the user is an admin in a specific chat."""
    try:
        # For private chats, always consider the user an admin if they're in ADMIN_USERS
        if chat_id == user_id:
            return user_id in ADMIN_USERS
            
        # For groups, check if user is an admin in that group
        chat_member = await context.bot.get_chat_member(chat_id, user_id)
        return user_id in ADMIN_USERS or chat_member.status in [ChatMember.ADMINISTRATOR, ChatMember.CREATOR]
    except Exception as e:
        logger.error(f"Error checking admin status: {str(e)}")
        return False
        
def admin_only(func):
    """Decorator to restrict command access to admin users only"""
    @wraps(func)
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        
        # Check if user is an admin
        if not await is_admin(context, chat_id, user_id):
            await update.message.reply_text("Sorry, this command is restricted to admins only.")
            return
            
        return await func(update, context, *args, **kwargs)
    return wrapped

def get_categories() -> List[str]:
    """Get the list of categories."""
    try:
        with open(CATEGORIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("categories", [])
    except Exception as e:
        logger.error(f"Error reading categories: {str(e)}")
        return ["General"]
        
def add_category(category: str) -> bool:
    """Add a new category."""
    if not category:
        return False
        
    categories = get_categories()
    if category in categories:
        return True
        
    categories.append(category)
    try:
        with open(CATEGORIES_FILE, "w", encoding="utf-8") as f:
            json.dump({"categories": categories}, f)
        return True
    except Exception as e:
        logger.error(f"Error writing categories: {str(e)}")
        return False

def read_entries(group_id: Optional[int] = None, category: Optional[str] = None) -> List[Dict[str, str]]:
    """Read entries from the CSV file with optional filtering by group_id and category."""
    entries = []
    try:
        with open(ENTRIES_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Fill in missing fields with defaults for backward compatibility
                if "category" not in row:
                    row["category"] = "General"
                if "group_id" not in row:
                    row["group_id"] = ""
                
                # Apply filters if specified
                if group_id is not None and row["group_id"] and int(row["group_id"]) != group_id:
                    continue
                if category is not None and row["category"] != category:
                    continue
                    
                entries.append(row)
    except Exception as e:
        logger.error(f"Error reading entries: {str(e)}")
    return entries

def write_entries(entries: List[Dict[str, str]]) -> bool:
    """Write entries to the CSV file."""
    try:
        with open(ENTRIES_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
            writer.writerows(entries)
        return True
    except Exception as e:
        logger.error(f"Error writing entries: {str(e)}")
        return False

def add_entry(text: str, link: str, category: str = "General", group_id: Optional[int] = None) -> bool:
    """Add a new entry to the CSV file."""
    entries = read_entries()
    
    # Check if entry already exists
    for entry in entries:
        if entry["text"] == text and entry["link"] == link:
            if (group_id is None or entry["group_id"] == str(group_id)):
                return False
    
    # Make sure category exists or create it
    add_category(category)
    
    new_entry = {
        "text": text, 
        "link": link, 
        "category": category,
        "group_id": str(group_id) if group_id else ""
    }
    
    entries.append(new_entry)
    return write_entries(entries)

def delete_entry(index: int) -> bool:
    """Delete an entry from the CSV file."""
    entries = read_entries()
    if 0 <= index < len(entries):
        entries.pop(index)
        return write_entries(entries)
    return False

def clear_all_entries(group_id: Optional[int] = None, category: Optional[str] = None) -> int:
    """Clear all entries, optionally filtered by group_id and/or category.
    Returns the number of entries deleted."""
    all_entries = read_entries()
    
    if group_id is None and category is None:
        # Clear all entries
        count = len(all_entries)
        return count if write_entries([]) else 0
    
    # Filter entries to keep
    entries_to_keep = []
    count = 0
    
    for entry in all_entries:
        should_delete = True
        
        if group_id is not None and entry["group_id"] and int(entry["group_id"]) != group_id:
            should_delete = False
        if category is not None and entry["category"] != category:
            should_delete = False
            
        if should_delete:
            count += 1
        else:
            entries_to_keep.append(entry)
    
    if count > 0:
        success = write_entries(entries_to_keep)
        return count if success else 0
    return 0

def search_entries(query: str, group_id: Optional[int] = None, category: Optional[str] = None) -> List[Dict[str, str]]:
    """Search for entries matching the query, with optional group_id and category filtering."""
    # Get entries with the specified group_id and category filters
    entries = read_entries(group_id=group_id, category=category)
    
    # If no query provided, just return filtered entries
    if not query:
        return entries
    
    # Search in both text and category fields
    query = query.lower()
    return [entry for entry in entries if 
            query in entry["text"].lower() or 
            query in entry.get("category", "").lower()]

# ---------------------- NEW: GROQ API RESPONSE ----------------------

async def async_generate_response(prompt: str) -> str:
    """Generate a response using the Groq API."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(GROQ_API_URL, headers=headers, json=data, timeout=60) as resp:
            if resp.status == 200:
                res = await resp.json()
                return res["choices"][0]["message"]["content"].strip()
            else:
                error_text = await resp.text()
                logger.error(f"Groq API error {resp.status}: {error_text}")
                raise Exception(f"Groq API error {resp.status}: {error_text}")

# ---------------------- COMMAND HANDLERS -------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    chat_id = update.effective_chat.id
    is_user_admin = await is_admin(context, chat_id, user.id)
    
    help_text = (
        f"👋 Hi {user.mention_html()}! I'm Summarizer2, an AI-powered bot for managing knowledge entries.\n\n"
        "Available commands:\n"
        "/ask <your question > - Ask a question about the stored entries\n"
        "/here <your question > - Answer a question (when replying to someone)\n"
    )
    
    if is_user_admin:
        admin_text = (
            "/list - List knowledge entries (admin only)\n"
            "/add \"entry text\" \"message_link\" \"category\" - Add a new entry\n"
            "/download - Download the current CSV file\n"
            "/upload - Upload a CSV file\n"
            "/clear - Clear all entries or entries in a specific category\n"
            "/logs - Show last 10 log entries\n"  # Add this line
        )
        help_text += admin_text
    
    help_text += "\nUse categories to organize your knowledge entries."
    await update.message.reply_html(help_text)

def build_prompt(question: str, context_text: str) -> str:
    """Build a prompt for the LLM."""
    prompt = (
        "You are an expert assistant. Use the following context to answer the question. "
        "Cite relevant entries and be concise.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return prompt

def add_hyperlinks(answer: str, keywords: Dict[str, str]) -> str:
    """Add hyperlinks to keywords in the answer if their text matches an entry."""
    for key, link in keywords.items():
        if key in answer:
            answer = answer.replace(key, f'<a href="{link}">{key}</a>')
    return answer

async def here_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Answer a question using the Groq API but reply to the person being replied to and delete the command."""
    # Check if this message is a reply
    if update.message.reply_to_message is None:
        await update.message.reply_text("This command must be used as a reply to another message.")
        return

    # Get the question from the command text
    command_text = update.message.text
    question = command_text[5:].strip()  # Remove "/here "

    if not question:
        await update.message.reply_text(
            "Please provide a question after the /here command. For example:\n"
            "/here What are the main features of Summarizer2?"
        )
        return

    # Get the message this is replying to
    replied_msg = update.message.reply_to_message
    replied_user = replied_msg.from_user

    # Send initial thinking message
    thinking_message = await update.message.reply_text("🤔 Thinking about your question... This might take a moment.")

    # Determine group ID for group-specific entries if in a group chat
    group_id = None
    if update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP]:
        group_id = update.effective_chat.id

    # Get entries for context - specific to this group if in a group chat
    entries = read_entries(group_id=group_id)

    if not entries:
        await thinking_message.delete()
        await replied_msg.reply_text(
            f"{replied_user.mention_html()}, no knowledge entries found to answer your question.", 
            parse_mode=ParseMode.HTML
        )
        await update.message.delete()
        return

    try:
        await thinking_message.edit_text("🤔")
        # Create context from entries with categories
        context_entries = []
        for entry in entries:
            category = entry.get("category", "General")
            entry_text = f"Category: {category}\nEntry: {entry['text']}\nSource: {entry['link']}"
            context_entries.append(entry_text)
        context_text = "\n\n".join(context_entries)
        prompt = build_prompt(question, context_text)

        await thinking_message.edit_text("⚡")
        answer = await async_generate_response(prompt)

        # Add hyperlinks to the answer
        keywords = {entry["text"]: entry["link"] for entry in entries}
        final_answer = add_hyperlinks(answer, keywords)

        # Send response to the replied user
        await thinking_message.delete()
        await replied_msg.reply_text(
            f"{replied_user.mention_html()}, here's the answer to: {question}\n\n{final_answer}",
            parse_mode=ParseMode.HTML
        )

        # Delete the original command message
        try:
            await update.message.delete()
        except Exception as e:
            logger.error(f"Error deleting message: {str(e)}")

    except Exception as e:
        logger.error(f"Error generating LLM response: {str(e)}")
        await thinking_message.delete()
        await replied_msg.reply_text(
            f"{replied_user.mention_html()}, sorry, I encountered an error while processing your question.\n"
            f"Error: {str(e)[:100]}...",
            parse_mode=ParseMode.HTML
        )
        # Still try to delete the command message
        try:
            await update.message.delete()
        e
