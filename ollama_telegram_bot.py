import os
import csv
import sys
from storage import EntryStorage
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
import nest_asyncio
import threading
import asyncio
import time
from typing import Callable, Optional, Dict, Tuple
from apscheduler.schedulers.background import BackgroundScheduler
# Apply nest_asyncio to patch the event loop
nest_asyncio.apply()
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
import requests
import json
from flask import Flask, jsonify
import logging.handlers
from telegram import InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import CommandHandler, CallbackQueryHandler
# Groq API client will be imported as needed
# Search is now handled by SQLite FTS5 through storage
# No need for duplicate imports as they're already defined above

# Global variable for slowmode seconds (default)
SLOWMODE_SECONDS = 3  # Default, can be changed via /slowmode command

# Per-user per-command last called tracking (in-memory)
_user_command_timestamps: Dict[Tuple[int, str], float] = {}

def set_slowmode(seconds: int):
    global SLOWMODE_SECONDS
    SLOWMODE_SECONDS = max(1, int(seconds))

def get_slowmode():
    return SLOWMODE_SECONDS

def rate_limit(key_func: Optional[Callable] = None):
    """
    Decorator to limit command usage per user; deletes the message if rate limited.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(update, context, *args, **kwargs):
            user_id = update.effective_user.id if update.effective_user else None
            command = func.__name__
            key = (user_id, command)
            if key_func:
                key = key_func(update, context)
            now = time.monotonic()
            last = _user_command_timestamps.get(key, 0)
            limit_seconds = get_slowmode()
            if now - last < limit_seconds:
                # Attempt to delete the triggering message
                try:
                    if hasattr(update, "message") and update.message and update.message.delete:
                        await update.message.delete()
                        logger.info(f"Rate limited and deleted message from user_id {user_id} for command '{command}'.")
                except Exception as e:
                    logger.warning(f"Failed to delete message for rate-limited user_id {user_id} on command '{command}': {e}")
                # Silently ignore, do not send any message
                return
            _user_command_timestamps[key] = now
            return await func(update, context, *args, **kwargs)
        return wrapper
    return decorator

# --- SQLite Storage Layer (add this section before your handlers, after imports) ---
import sqlite3
import threading

class EntryStorage:
    def __init__(self, db_path="entries.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        with self._get_conn() as conn:
            c = conn.cursor()
            c.execute('CREATE VIRTUAL TABLE IF NOT EXISTS entries USING FTS5(text, link, category)')
            c.execute('CREATE TABLE IF NOT EXISTS categories (name TEXT PRIMARY KEY)')
            c.execute('SELECT COUNT(*) FROM categories')
            if c.fetchone()[0] == 0:
                c.executemany('INSERT INTO categories (name) VALUES (?)',
                              [('General',), ('Documentation',), ('Tutorials',), ('References',)])
            conn.commit()

    def add_entry(self, text: str, link: str, category: str = "General") -> bool:
        with self._lock, self._get_conn() as conn:
            c = conn.cursor()
            c.execute('SELECT 1 FROM entries WHERE text=? AND link=?', (text, link))
            if c.fetchone():
                return False
            c.execute('INSERT INTO entries (text, link, category) VALUES (?, ?, ?)', (text, link, category))
            c.execute('INSERT OR IGNORE INTO categories (name) VALUES (?)', (category,))
            conn.commit()
        return True

    def insert_entry_at(self, index: int, text: str, link: str, category: str = "General") -> bool:
        with self._lock, self._get_conn() as conn:
            c = conn.cursor()
            c.execute('SELECT text, link, category FROM entries ORDER BY rowid')
            entries = c.fetchall()
            new_row = (text, link, category)
            if index < 0 or index > len(entries):
                return False
            entries = entries[:index] + [new_row] + entries[index:]
            c.execute('DELETE FROM entries')
            c.executemany('INSERT INTO entries (text, link, category) VALUES (?, ?, ?)', entries)
            c.execute('INSERT OR IGNORE INTO categories (name) VALUES (?)', (category,))
            conn.commit()
        return True

    def get_entries(self, category: Optional[str] = None) -> list:
        with self._get_conn() as conn:
            c = conn.cursor()
            if category:
                c.execute('SELECT rowid, text, link, category FROM entries WHERE category=? ORDER BY rowid', (category,))
            else:
                c.execute('SELECT rowid, text, link, category FROM entries ORDER BY rowid')
            return [{"id": row[0], "text": row[1], "link": row[2], "category": row[3]} for row in c.fetchall()]

    def get_entry_by_index(self, index: int) -> Optional[dict]:
        entries = self.get_entries()
        if 0 <= index < len(entries):
            return entries[index]
        return None

    def delete_entry_by_index(self, index: int) -> bool:
        entries = self.get_entries()
        if 0 <= index < len(entries):
            entry_id = entries[index]['id']
            with self._lock, self._get_conn() as conn:
                c = conn.cursor()
                c.execute('DELETE FROM entries WHERE rowid=?', (entry_id,))
                conn.commit()
                return c.rowcount > 0
        return False

    def search_entries(self, query: str, category: Optional[str] = None, top_n: int = 8) -> list:
        with self._get_conn() as conn:
            c = conn.cursor()
            match_query = query
            if category:
                match_query = f'{query} category:{category}'
            c.execute('SELECT rowid, text, link, category FROM entries WHERE entries MATCH ? ORDER BY rank LIMIT ?', (match_query, top_n))
            return [{"id": row[0], "text": row[1], "link": row[2], "category": row[3]} for row in c.fetchall()]

    def clear_entries(self, category: Optional[str] = None) -> int:
        with self._lock, self._get_conn() as conn:
            c = conn.cursor()
            if category:
                c.execute('DELETE FROM entries WHERE category=?', (category,))
            else:
                c.execute('DELETE FROM entries')
            affected = c.rowcount
            conn.commit()
            return affected

    def get_categories(self) -> list:
        with self._get_conn() as conn:
            c = conn.cursor()
            c.execute('SELECT name FROM categories ORDER BY name')
            return [row[0] for row in c.fetchall()]

    def add_category(self, category: str) -> bool:
        with self._lock, self._get_conn() as conn:
            c = conn.cursor()
            try:
                c.execute('INSERT INTO categories (name) VALUES (?)', (category,))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

# Instantiate the storage object after this class
storage = EntryStorage()

# Add these constants at the top of your file
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
        """Send a message to logs channel."""
        app = Application.builder().token(self.bot_token).build()
        try:
            await app.bot.send_message(
                chat_id=-1001925908750,  # Logs channel ID
                text=message,
                parse_mode='HTML'
            )
        except Exception as e:
            logging.error(f"Failed to send status to logs channel: {e}")
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
BOT_TOKEN = "6614402193:AAGXg-AS9xZV8A7n6SHL0Wy2-dOstLu8FdI"
bot_token = BOT_TOKEN

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
ADMIN_USER_IDS = "5980915474, 2102500190, 6224005135, 5220713961, 1217170295"
ADMIN_USERS = [int(uid.strip()) for uid in ADMIN_USER_IDS.split(",") if uid.strip()]
CURRENT_DATE = "2025-04-27 09:19:30"  # Updated current UTC time
CURRENT_USER = "GuardianAngelWw"      # Updated current user
ENTRIES_FILE = "entries.csv"
CATEGORIES_FILE = "categories.json"
CSV_HEADERS = ["text", "link", "category"]  # Removed group_id

# Update the model configuration for Groq API
TOGETHER_API_KEY = os.getenv("GROQ_API_KEY", "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")
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # Using Groq compatible model

# Global mutable configuration for runtime updates via admin commands
CURRENT_AI_MODEL = GROQ_MODEL
CURRENT_AI_API_KEY = TOGETHER_API_KEY

def set_ai_model(new_model: str) -> bool:
    global CURRENT_AI_MODEL
    if not new_model or not isinstance(new_model, str):
        return False
    CURRENT_AI_MODEL = new_model.strip()
    logger.info(f"AI model updated to: {CURRENT_AI_MODEL}")
    return True

def set_ai_api_key(new_key: str) -> bool:
    global CURRENT_AI_API_KEY
    if not new_key or not isinstance(new_key, str):
        return False
    CURRENT_AI_API_KEY = new_key.strip()
    logger.info("AI API key updated (not shown for security).")
    return True

# Flask app initialization
app = Flask(__name__)

# Modify the startup logging to be more secure (around line 72)
logger.info(f"Bot starting at {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
logger.info(f"Using Groq API with model {GROQ_MODEL}")
logger.info("Bot initialization successful")  # Instead of logging the token

# Flask routes for health monitoring
@app.route('/health')
def health_check():
    """Health check endpoint for container monitoring"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.utcnow().isoformat(),
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

# Log the model loading
logger.info(f"Bot started with TOGETHER API")

# Pagination configuration
ENTRIES_PER_PAGE = 5

# Update the initialization of the CSV file (remove group_id)
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
        return user_id in ADMIN_USERS or chat_member.status in ["administrator", "creator"]
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

@admin_only
async def insert_entry_command(update, context):
    text = update.message.text
    match = re.match(r'/i\s+(\d+)\s+"([^"]*(?:\\"[^"]*)*?)"\s+"([^"]*(?:\\"[^"]*)*?)"(?:\s+"([^"]*(?:\\"[^"]*)*?)")?', text)
    if not match:
        await update.message.reply_text('Usage:\n/i <row_number> "entry text" "link" "optional_category"\nExample:\n/i 3 "Some text" "https://example.com" "Category"')
        return
    row = int(match.group(1)) - 1
    entry_text = match.group(2)
    link = match.group(3)
    category = match.group(4) if match.group(4) else "General"
    if storage.insert_entry_at(row, entry_text, link, category):
        await update.message.reply_text(f"✅ Inserted at row {row+1}:\nCategory: {category}\nText: {entry_text}\nLink: {link}")
    else:
        await update.message.reply_text("❌ Error: Invalid row or duplicate entry.")

@admin_only
async def show_entry_command(update, context):
    args = context.args
    if not args or not args[0].isdigit():
        await update.message.reply_text("Usage: /s <entry_number>\nExample: /s 4")
        return
    idx = int(args[0]) - 1
    entry = storage.get_entry_by_index(idx)
    if not entry:
        await update.message.reply_text(f"Entry #{args[0]} does not exist.")
        return
    msg = (f"<b>Entry #{idx+1}</b>\n"
           f"<b>Category:</b> {entry['category']}\n"
           f"<b>Text:</b>\n<blockquote>{entry['text']}</blockquote>\n"
           f"<b>Link:</b> <a href='{entry['link']}'>{entry['link']}</a>")
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🗑️ Delete", callback_data=f"sdelete:{idx}")]
    ])
    await update.message.reply_html(msg, reply_markup=keyboard, disable_web_page_preview=True)

@admin_only
async def handle_single_entry_delete(update, context):
    query = update.callback_query
    idx = int(query.data.split(":", 1)[1])
    if storage.delete_entry_by_index(idx):
        await query.edit_message_text(f"✅ Entry #{idx+1} deleted successfully.")
    else:
        await query.answer("Failed to delete entry.", show_alert=True)

@admin_only
async def set_model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /setmodel <model-name>")
        return
    new_model = " ".join(context.args).strip()
    if set_ai_model(new_model):
        await update.message.reply_text(
            f"✅ AI model updated to: <code>{CURRENT_AI_MODEL}</code>",
            parse_mode=ParseMode.HTML
        )
    else:
        await update.message.reply_text("❌ Failed to update model. Provide a valid model name.")

@admin_only
async def set_apikey_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Never echo back the API key!
    if not context.args:
        await update.message.reply_text("Usage: /setapikey <API-key>")
        return
    new_key = " ".join(context.args).strip()
    if set_ai_api_key(new_key):
        await update.message.reply_text("✅ AI API key updated successfully.")
    else:
        await update.message.reply_text("❌ Failed to update API key. Provide a valid key.")

# ---- Slowmode control command (admins only) ----

@admin_only
async def slowmode_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Set the global slowmode (rate limit) in seconds. Usage: /slowmode 10
    """
    if context.args and context.args[0].isdigit():
        seconds = int(context.args[0])
        set_slowmode(seconds)
   
