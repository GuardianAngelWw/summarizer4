import os
import csv
import sys
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

# --- Add command logging decorator function ---
def log_command(func):
    @wraps(func)
    async def wrapper(update, context, *args, **kwargs):
        user = update.effective_user
        if not user:
            logger.warning(f"Command received with no effective_user")
            return await func(update, context, *args, **kwargs)
            
        username = f"@{user.username}" if user.username else f"{user.first_name}"
        user_id = user.id
        command = update.message.text if update.message else "(callback)"
        
        logger.info(f"Command received: {command} from user {username} (ID: {user_id})")
        
        try:
            return await func(update, context, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing command {command} from {username}: {str(e)}", exc_info=True)
            # Try to notify the user about the error
            try:
                if update.message and hasattr(update.message, 'reply_text'):
                    await update.message.reply_text(f"⚠️ Error processing your command: {str(e)}")
            except Exception:
                pass  # If we can't reply, just log and continue
            raise
            
    return wrapper

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

    def clear_entries(self, category: Optional[str] = None) -> bool:
        with self._lock, self._get_conn() as conn:
            c = conn.cursor()
            if category:
                c.execute('DELETE FROM entries WHERE category=?', (category,))
            else:
                c.execute('DELETE FROM entries')
            conn.commit()
        return True

    def delete_entry_by_index(self, index: int) -> bool:
        entries = self.get_entries()
        if 0 <= index < len(entries):
            entry_id = entries[index]['id']
            with self._lock, self._get_conn() as conn:
                c = conn.cursor()
                c.execute('DELETE FROM entries WHERE rowid=?'', (entry_id,))
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
# Load environment variables
load_dotenv()

# Get bot token from environment variable, with a fallback for backward compatibility
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "6614402193:AAH6s6YLRCY0bnjT1-844FuGYJvjhkUyFDg")
bot_token = BOT_TOKEN

# Check if token is available
if not BOT_TOKEN:
    logging.error("TELEGRAM_BOT_TOKEN environment variable not set. Please set it in your .env file.")
    sys.exit(1)

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
async def download_db(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    db_path = "entries.db"
    if not os.path.exists(db_path):
        await update.message.reply_text("No entries database found.")
        return
    await update.message.reply_document(
        document=open(db_path, "rb"),
        filename="entries.db",
        caption="Here's your complete knowledge base database file."
    )

@admin_only
async def handle_db_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message.reply_to_message and update.message.document:
        context.user_data["awaiting_db"] = True
    if not context.user_data.get("awaiting_db"):
        return
    context.user_data["awaiting_db"] = False
    
    if not update.message.document:
        await update.message.reply_text("Please upload an SQLite database file (.db).")
        return
        
    document = update.message.document
    if not document.file_name.lower().endswith(".db"):
        await update.message.reply_text("Please upload a file ending with .db")
        return
        
    processing_status = await update.message.reply_text("⏳ Processing your uploaded database file...")
    
    try:
        file = await context.bot.get_file(document.file_id)
        db_path = "entries.db"
        backup_path = db_path + ".backup"
        temp_path = db_path + ".upload"
        
        # Download the uploaded file
        await file.download_to_drive(temp_path)
        
        # Verify the uploaded file is a valid SQLite database with expected schema
        try:
            conn = sqlite3.connect(temp_path)
            cursor = conn.cursor()
            
            # Check if it has the expected tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' OR type='virtual table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            if 'entries' not in tables or 'categories' not in tables:
                await processing_status.edit_text(
                    "❌ The uploaded file is not a valid entries database. "
                    "It's missing required tables (entries or categories)."
                )
                conn.close()
                os.remove(temp_path)
                return
                
            # Check if entries has the expected columns
            try:
                cursor.execute("SELECT rowid, text, link, category FROM entries LIMIT 1")
                conn.close()
                
                # Create backup of current database
                if os.path.exists(db_path):
                    shutil.copy2(db_path, backup_path)
                    
                # Replace the current database
                os.replace(temp_path, db_path)
                
                # Reload the storage to use the new database
                storage._init_db()
                
                # Get entry count for feedback
                new_entries = storage.get_entries()
                category_counts = {}
                for entry in new_entries:
                    cat = entry.get('category', 'General')
                    category_counts[cat] = category_counts.get(cat, 0) + 1
                
                category_info = "\n".join([f"- {cat}: {count} entries" for cat, count in category_counts.items()])
                
                await processing_status.edit_text(
                    f"✅ Database file successfully uploaded and applied!\n"
                    f"Total entries: {len(new_entries)}\n\n"
                    f"Entries by category:\n{category_info}"
                )
                
            except sqlite3.Error:
                conn.close()
                await processing_status.edit_text(
                    "❌ The uploaded database doesn't have the expected structure."
                )
                os.remove(temp_path)
                
        except sqlite3.Error as e:
            await processing_status.edit_text(f"❌ Invalid SQLite database file: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        logger.error(f"Error processing database upload: {str(e)}")
        try:
            await processing_status.edit_text(f"❌ Error processing the uploaded file: {str(e)}")
        except:
            await update.message.reply_text(f"❌ Error processing the uploaded file: {str(e)}")
        
        if os.path.exists(temp_path):
            os.remove(temp_path)

async def send_db_to_logs_channel(bot_token: str, file_path: str, channel_id: int):
    """Send the SQLite DB file to the specified Telegram channel."""
    try:
        app = Application.builder().token(bot_token).build()
        await app.bot.send_document(
            chat_id=channel_id,
            document=open(file_path, "rb"),
            filename=os.path.basename(file_path),
            caption="📦 Daily #backup: Current entries.db file."
        )
        await app.shutdown()
        logger.info("Successfully sent daily DB backup to logs channel.")
    except Exception as e:
        logger.error(f"Error sending DB to logs channel: {str(e)}")

def schedule_daily_db_backup(bot_token: str, file_path: str, channel_id: int):
    """Schedule sending the DB file to the logs channel once every day."""
    scheduler = BackgroundScheduler(timezone="UTC")
    async def send_backup():
        await send_db_to_logs_channel(bot_token, file_path, channel_id)
    def job():
        try:
            asyncio.run(send_backup())
        except Exception as e:
            logger.error(f"Error in scheduled DB backup: {e}")
    scheduler.add_job(job, "cron", hour=0, minute=10, id="daily_db_backup", replace_existing=True)
    scheduler.start()
    logger.info("Scheduled daily DB backup to logs channel.")

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
        await update.message.reply_text(f"⏱ Slow mode set to {seconds} seconds.")
        logger.info(f"Slow mode set to {seconds} seconds by admin {update.effective_user.id}.")
    else:
        await update.message.reply_text(
            f"Usage: /slowmode <seconds>\nCurrent: {get_slowmode()} seconds.")


def clean_telegram_html(text: str) -> str:
    """
    Clean/sanitize string for Telegram HTML compatibility:
    - Replace <br>, <br/>, </br> with newlines.
    - Remove all other unsupported tags.
    - Optionally, collapse multiple newlines.
    """
    # Replace <br> and variants with newline
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</br\s*>', '\n', text, flags=re.IGNORECASE)

    # Allowed Telegram HTML tags
    allowed = ['b','strong','i','em','u','ins','s','strike','del','span','tg-spoiler','a','code','pre','blockquote']
    # Remove all other HTML tags except allowed
    text = re.sub(
        r'</?(?!' + '|'.join(allowed) + r')\b[^>]*>',
        '',
        text
    )

    # Collapse >2 newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text

# ... [rest of your code remains the same, but update message sending as follows] ...

async def send_csv_to_logs_channel(bot_token: str, file_path: str, channel_id: int):
    """Send the CSV file to the specified Telegram channel."""
    try:
        app = Application.builder().token(bot_token).build()
        await app.bot.send_document(
            chat_id=channel_id,
            document=open(file_path, "rb"),
            filename=os.path.basename(file_path),
            caption="📦 Daily #backup: Current entries.csv file."
        )
        await app.shutdown()
        logger.info("Successfully sent daily CSV backup to logs channel.")
    except Exception as e:
        logger.error(f"Error sending CSV to logs channel: {str(e)}")

def schedule_daily_csv_backup(bot_token: str, file_path: str, channel_id: int):
    """Schedule sending the CSV file to the logs channel once every day."""
    scheduler = BackgroundScheduler(timezone="UTC")

    async def send_backup():
        await send_csv_to_logs_channel(bot_token, file_path, channel_id)

    def job():
        try:
            asyncio.run(send_backup())
        except Exception as e:
            logger.error(f"Error in scheduled CSV backup: {e}")

    # Schedule for once every day at 00:10 UTC (can adjust the time as needed)
    scheduler.add_job(job, "cron", hour=0, minute=10, id="daily_csv_backup", replace_existing=True)
    scheduler.start()
    logger.info("Scheduled daily CSV backup to logs channel.")

'''def get_categories() -> List[str]:
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
        return False '''

'''def read_entries(category: Optional[str] = None) -> List[Dict[str, str]]:
    """Read entries from the CSV file with optional filtering by category only."""
    entries = []
    try:
        with open(ENTRIES_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Fill in missing fields
                if "category" not in row:
                    row["category"] = "General"
                # Remove group_id if present from previous versions
                row.pop("group_id", None)
                if category is not None and row["category"] != category:
                    continue
                entries.append(row)
    except Exception as e:
        logger.error(f"Error reading entries: {str(e)}")
    return entries

def write_entries(entries: List[Dict[str, str]]) -> bool:
    """Write entries to the CSV file (category only, no group_id)."""
    try:
        with open(ENTRIES_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
            writer.writerows(entries)
        return True
    except Exception as e:
        logger.error(f"Error writing entries: {str(e)}")
        return False

def add_entry(text: str, link: str, category: str = "General") -> bool:
    """Add a new entry (no group-specific logic)."""
    entries = read_entries()
    for entry in entries:
        if entry["text"] == text and entry["link"] == link:
            return False
    add_category(category)
    new_entry = {
        "text": text,
        "link": link,
        "category": category
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

def clear_all_entries(category: Optional[str] = None) -> int:
    """Clear all entries, optionally filtered by category only."""
    all_entries = read_entries()
    if category is None:
        count = len(all_entries)
        return count if write_entries([]) else 0
    entries_to_keep = []
    count = 0
    for entry in all_entries:
        if entry["category"] != category:
            entries_to_keep.append(entry)
        else:
            count += 1
    if count > 0:
        success = write_entries(entries_to_keep)
        return count if success else 0
    return 0 '''

# Search functionality is now handled by storage.search_entries using SQLite FTS5

def search_entries(query: str, category: Optional[str] = None) -> List[Dict[str, str]]:
    """Search for entries matching the query, with optional category filtering (no group)."""
    entries = read_entries(category=category)
    if not query:
        return entries
    query = query.lower()
    return [entry for entry in entries if 
            query in entry["text"].lower() or 
            query in entry.get("category", "").lower()]

# Updated load_llm function to use Groq API
async def load_llm():
    try:
        logger.info(f"Using Groq API with model: {CURRENT_AI_MODEL}")
        if not CURRENT_AI_API_KEY:
            logger.error("AI API key is not set. Please set it with /setapikey.")
            raise ValueError("AI API key is required")
        return {"groq_client": True}
    except Exception as e:
        logger.error(f"Error initializing Groq client: {str(e)}")
        raise

def get_context_for_question(question: str, category: Optional[str] = None, top_n: int = 8) -> str:
    """
    Build context string from most relevant entries for a question.
    """
    relevant_entries = storage.search_entries(question, category, top_n)
    return "\n\n".join(
        f"Category: {entry.get('category', 'General')}\nEntry: {entry['text']}\nSource: {entry['link']}"
        for entry in relevant_entries
    )

# Command Handlers
@log_command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    chat_id = update.effective_chat.id
    is_user_admin = await is_admin(context, chat_id, user.id)
    
    help_text = (
        f"👋 Hi {user.mention_html()}! I'm here to help you.\n\n"
        "Available commands:\n"
        "/ask &lt;your question &gt; - Ask a question about wolfblood and networks\n"
        "/here &lt;your question &gt; - Ask a question (when replying to someone)\n"
    )
    
#    if is_user_admin:
#       admin_text = (
#            "/list - List knowledge entries (admin only)\n"
#            "/add \"entry text\" \"message_link\" \"category\" - Add a new entry\n"
#            "/download - Download the current CSV file\n"
#            "/upload - Upload a CSV file\n"
#            "/clear - Clear all entries or entries in a specific category\n"
#        )
#        help_text += admin_text
    
 #   help_text += "\nUse categories to organize your knowledge entries."
    await update.message.reply_html(help_text)

@rate_limit()
@log_command
async def here_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Answer a question using the LLM but reply to the person being replied to and delete the command."""
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
            "/here What are some betrayal cases in wolfblood?"
        )
        return
    
    # Get the message this is replying to
    replied_msg = update.message.reply_to_message
    replied_user = replied_msg.from_user
    
    # Send initial thinking message
    thinking_message = await update.message.reply_text("🤔 Thinking about your question... This might take a moment.")
    
    context_text = get_context_for_question(question, top_n=8)
    if not context_text.strip():
        await thinking_message.delete()
        await replied_msg.reply_text(
            f"{replied_user.mention_html()}, no knowledge entries found to answer your question.", 
            parse_mode=ParseMode.HTML
        )
        await update.message.delete()
        return

    try:
        await thinking_message.edit_text("🤔")
        await load_llm()
        await thinking_message.edit_text("⚡")
        prompt = build_prompt(question, context_text)

        # Build keywords from relevant entries for hyperlinks
        relevant_entries = storage.search_entries(question, top_n=8)
        keywords = {entry["text"]: entry["link"] for entry in relevant_entries}
        answer = await generate_response(prompt, None, None)
        final_answer = add_hyperlinks(answer, keywords)

        await thinking_message.delete()
        if len(final_answer) > 4000:
            final_answer = final_answer[:3900] + "\n\n... (message truncated due to length)"
        try:
            await replied_msg.reply_text(
                f"{replied_user.mention_html()} 👇 {clean_telegram_html(final_answer)}",
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True
            )
        except Exception as e:
            logger.error(f"Error sending response: {str(e)}")
            await replied_msg.reply_text(
                f"{replied_user.mention_html()}, fool !! I'm 𝘯𝘰𝘵 𝘺𝘰𝘶𝘳 𝘴𝘦𝘳𝘷𝘢𝘯𝘵!",
                parse_mode=ParseMode.HTML
            )
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
        try:
            await update.message.delete()
        except Exception as e:
            logger.error(f"Error deleting message: {str(e)}")

# Add the logs command handler
@admin_only
@log_command
async def show_logs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send the last 10 log entries to the chat."""
    if not last_logs:
        await update.message.reply_text("No logs available.")
        return

    # Format logs for display
    log_text = "📋 Last 10 log entries:\n\n"
    for log in last_logs:
        log_text += f"{log}\n"

    # Split message if it's too long
    if len(log_text) > 4000:  # Telegram message limit is 4096 characters
        parts = [log_text[i:i+4000] for i in range(0, len(log_text), 4000)]
        for part in parts:
            await update.message.reply_text(part)
    else:
        await update.message.reply_text(log_text)

# Add custom error handler to exclude sensitive data
def format_error_for_user(error: Exception) -> str:
    """Format error message for user, excluding sensitive information."""
    error_str = str(error)
    # List of patterns to remove/replace
    sensitive_patterns = [
        (r'token=[a-zA-Z0-9:_-]+', 'token=<REDACTED>'),
        (r'api_key=[a-zA-Z0-9_-]+', 'api_key=<REDACTED>'),
        (r'password=[a-zA-Z0-9@#$%^&*]+', 'password=<REDACTED>'),
        (r'BOT_TOKEN=[a-zA-Z0-9:_-]+', 'BOT_TOKEN=<REDACTED>')
    ]
    
    for pattern, replacement in sensitive_patterns:
        error_str = re.sub(pattern, replacement, error_str)
    return error_str

@admin_only
async def add_entry_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text[5:].strip()
    match = re.match(r'"([^"]*(?:\\"[^"]*)*?)"\s+"([^"]*(?:\\"[^"]*)*?)"(?:\s+"([^"]*(?:\\"[^"]*)*?)")?', text)
    if not match:
        await update.message.reply_text(
            "Please use the format: /add \"entry text\" \"message_link\" \"optional_category\""
        )
        return
    match_groups = match.groups()
    entry_text = match_groups[0]
    link = match_groups[1]
    category = match_groups[2] if len(match_groups) > 2 and match_groups[2] else "General"
    if storage.add_entry(entry_text, link, category):
        await update.message.reply_text(f"✅ Added new entry:\n\nCategory: {category}\nText: {entry_text}\nLink: {link}")
    else:
        await update.message.reply_text("❌ Error: Entry already exists or could not be added.")

@admin_only
async def list_entries(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args if context.args else []
    query = ""
    category = None
    for arg in args:
        if arg.startswith("category="):
            category = arg.split("=")[1] if len(arg.split("=")) > 1 else None
        else:
            query = arg
    page = int(context.user_data.get('page', 0))
    entries = storage.get_entries(category)
    total_pages = (len(entries) + ENTRIES_PER_PAGE - 1) // ENTRIES_PER_PAGE
    start_idx = page * ENTRIES_PER_PAGE
    end_idx = min(start_idx + ENTRIES_PER_PAGE, len(entries))
    if not entries:
        message = "No entries found."
        if category:
            message += f" in category '{category}'"
        if query:
            message += f" matching '{query}'"
        await update.message.reply_text(message)
        return
    message = f"📚 Entries {start_idx+1}-{end_idx} of {len(entries)}"
    if category:
        message += f" in category '{category}'"
    if query:
        message += f" matching '{query}'"
    message += ":\n\n"
    for i, entry in enumerate(entries[start_idx:end_idx], start=start_idx + 1):
        message += f"{i}. [{entry.get('category', 'General')}] {entry['text']}\n"
        message += f"   🔗 {entry['link']}\n\n"
    
    # Create navigation buttons
    keyboard = []
    
    # Add category filter buttons
    categories = storage.get_categories()
    category_buttons = []
    for cat in categories[:3]:  # Limit to 3 buttons per row
        category_buttons.append(InlineKeyboardButton(
            f"📂 {cat}", 
            callback_data=f"cat:{cat}"
        ))
    
    if category_buttons:
        keyboard.append(category_buttons)
    
    # Add navigation buttons
    nav_row = []
    if page > 0:
        nav_row.append(InlineKeyboardButton("◀️ Previous", callback_data=f"page:{page-1}:{category or ''}"))
    
    if page < total_pages - 1:
        nav_row.append(InlineKeyboardButton("Next ▶️", callback_data=f"page:{page+1}:{category or ''}"))
    
    if nav_row:
        keyboard.append(nav_row)
    
    # Add delete buttons
    for i in range(start_idx, end_idx):
        keyboard.append([InlineKeyboardButton(
            f"🗑️ Delete #{i+1}", 
            callback_data=f"delete:{i}"
        )])
    
    # Add clear all button if there are entries
    if entries:
        clear_text = "Clear All"
        if category:
            clear_text = f"Clear '{category}' Entries"
        keyboard.append([InlineKeyboardButton(
            f"🗑️ {clear_text}", 
            callback_data=f"clear:{category or 'all'}"
        )])
    
    reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
    
    # Store the current page and category in user data
    context.user_data['page'] = page
    context.user_data['category'] = category
    
    # ********** PATCH STARTS HERE **********
    # Telegram message limit is 4096, use 4000 as a safe limit for buttons, etc.
    MAX_LEN = 4000
    if len(message) > MAX_LEN:
        logger.warning("Entry list message too long, splitting into multiple messages.")
        # Split at 4000 chars, but try not to break in the middle of a line
        lines = message.split('\n')
        chunk = ""
        for line in lines:
            if len(chunk) + len(line) + 1 > MAX_LEN:
                await update.message.reply_text(chunk, reply_markup=reply_markup)
                chunk = ""
            chunk += line + '\n'
        if chunk:
            await update.message.reply_text(chunk, reply_markup=reply_markup)
    else:
        await update.message.reply_text(message, reply_markup=reply_markup)
    # ********** PATCH ENDS HERE **********

# Update the handle_pagination function to check for admin permissions (around line 557)
async def handle_pagination(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle pagination callbacks and other inline button actions."""
    query = update.callback_query
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    
    # Always verify admin permissions first
    data = query.data

    # Only restrict certain actions to admins
    if data.startswith(("delete:", "clear:", "confirm_clear:")):
        is_user_admin = await is_admin(context, chat_id, user_id)
        if not is_user_admin:
            await query.answer("Sorry, only admins can use these controls.", show_alert=True)
            return

    await query.answer()
    
    if data.startswith("cat:"):
        category = data.split(":")[1]
        context.user_data['page'] = 0
        context.user_data['category'] = category
        fake_update = Update(update.update_id, message=update.effective_message)
        await list_entries(fake_update, context)
        return
    if data.startswith("page:"):
        parts = data.split(":")
        page = int(parts[1])
        category = parts[2] if len(parts) > 2 and parts[2] else None
        context.user_data['page'] = page
        entries = read_entries(category)
        
        # Calculate pagination
        total_pages = (len(entries) + ENTRIES_PER_PAGE - 1) // ENTRIES_PER_PAGE
        start_idx = page * ENTRIES_PER_PAGE
        end_idx = min(start_idx + ENTRIES_PER_PAGE, len(entries))
        
        # Build header message
        message = f"📚 Entries {start_idx+1}-{end_idx} of {len(entries)}"
        if category:
            message += f" in category '{category}'"
        message += ":\n\n"
        
        # Add entries to message with length limit check
        message_len = len(message)
        max_len = 3800  # Leave room for markup and footer
        
        for i, entry in enumerate(entries[start_idx:end_idx], start=start_idx + 1):
            entry_text = entry.get('text', '').strip()
            category_text = entry.get('category', 'General')
            link_text = entry.get('link', '').strip()
            
            # Truncate entry text if it's too long
            if len(entry_text) > 100:
                entry_text = entry_text[:97] + "..."
                
            # Truncate link if it's too long
            if len(link_text) > 60:
                link_text = link_text[:57] + "..."
                
            entry_message = f"{i}. [{category_text}] {entry_text}\n   🔗 {link_text}\n\n"
            
            # Check if adding this entry would exceed message length
            if message_len + len(entry_message) > max_len:
                message += "\n(Some entries truncated due to message length limit)"
                break
                
            message += entry_message
            message_len += len(entry_message)
        
        # Create navigation buttons
        keyboard = []
        
        # Add category filter buttons (only for admins)
        categories = storage.get_categories()
        category_buttons = []
        for cat in categories[:3]:  # Limit to 3 buttons per row
            category_buttons.append(InlineKeyboardButton(
                f"📂 {cat}", 
                callback_data=f"cat:{cat}"
            ))
        
        if category_buttons:
            keyboard.append(category_buttons)
        
        # Add navigation buttons
        nav_row = []
        if page > 0:
            nav_row.append(InlineKeyboardButton("◀️ Previous", callback_data=f"page:{page-1}:{category or ''}"))
        
        if page < total_pages - 1:
            nav_row.append(InlineKeyboardButton("Next ▶️", callback_data=f"page:{page+1}:{category or ''}"))
        
        if nav_row:
            keyboard.append(nav_row)
        
        # Add delete buttons (only shown to admins)
        for i in range(start_idx, end_idx):
            keyboard.append([InlineKeyboardButton(
                f"🗑️ Delete #{i+1}", 
                callback_data=f"delete:{i}"
            )])
        
        # Add clear all button if there are entries (only shown to admins)
        if entries:
            clear_text = "Clear All"
            if category:
                clear_text = f"Clear '{category}' Entries"
            keyboard.append([InlineKeyboardButton(
                f"🗑️ {clear_text}", 
                callback_data=f"clear:{category or 'all'}"
            )])
        
        reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
        
        # Update message with error handling
        try:
            await query.edit_message_text(message, reply_markup=reply_markup)
        except Exception as e:
            logger.error(f"Error updating message: {str(e)}")
            # If message is too long, try with a simpler message
            try:
                simple_message = f"📚 Showing entries for {category or 'all categories'}\n(Message simplified due to length issues)"
                await query.edit_message_text(simple_message, reply_markup=reply_markup)
            except Exception as e2:
                logger.error(f"Failed to send simplified message: {str(e2)}")
        return

    elif data.startswith("delete:"):
        index = int(data.split(":")[1])
        if delete_entry(index):
            await query.edit_message_text(f"✅ Entry #{index+1} deleted successfully.")
        else:
            await query.edit_message_text(f"❌ Failed to delete entry #{index+1}.")
    elif data.startswith("clear:"):
        category_filter = data.split(":")[1]
        category = None if category_filter == 'all' else category_filter
        confirm_text = "Are you sure you want to clear "
        if category:
            confirm_text += f"all entries in category '{category}'?"
        else:
            confirm_text += "ALL entries?"
        keyboard = [
            [
                InlineKeyboardButton("Yes, Clear", callback_data=f"confirm_clear:{category or 'all'}"),
                InlineKeyboardButton("Cancel", callback_data="cancel_clear")
            ]
        ]
        await query.edit_message_text(confirm_text, reply_markup=InlineKeyboardMarkup(keyboard))
    elif data.startswith("confirm_clear:"):
        category_filter = data.split(":")[1]
        category = None if category_filter == 'all' else category_filter
        count = clear_all_entries(category)
        if count > 0:
            await query.edit_message_text(f"✅ Successfully cleared {count} entries.")
        else:
            await query.edit_message_text("❌ No entries were cleared or an error occurred.")
    elif data == "cancel_clear":
        await query.edit_message_text("Operation cancelled.")

# Helper functions for ask_question
def build_prompt(question: str, context_text: str) -> str:
    return f"""You are an AI assistant with relatively high egoistic nature. Based on the provided knowledge base, summarize the context and provide a solution to the question without repeating it in the response:
    - While answering, ensure the answer is concise, understandable, and easy to follow and in very short brief.
    - **NEVER** send what you're **thinking** just send the processed output mentioned as below.
    - **MUST** use Telegram HTML markdown formatting for every answer with <blockquote>from starting of the answer to the end of the answer</blockquote>
    - **MUST** use Telegram HTML markdown formatting for every answer with <a href="source link">Relevant word of the output</a>.
    - **MUST** If the question contains any NSFW-themed content (violence, pornography, profanity, nudity, slurs, or other potentially disturbing subject matter), reply with "/report WB POLICE 🚓🚨🚔🚨🚓" as your output in telegram blockquote markdown format.
    - **MUST** read the whole question so every word of the question makes sense in the output.
    - **NEVER** mention about the knowledge base in the output or anything if you can / can't find.
    - **NEVER** reply out-of-context or out of entries questions.

    Question: {question}

    Knowledge Base:
    {context_text}"""


def add_hyperlinks(answer: str, keywords: Dict[str, str]) -> str:
    """
    Replace keywords with Telegram HTML links in the answer.

    :param answer: The generated answer text.
    :param keywords: A dictionary of keywords and their corresponding URLs.
    :return: Updated answer with hyperlinks.
    """
    def escape_html(text):
        return (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
        )

    for word, url in keywords.items():
        # Escape HTML in the word and URL
        safe_word = escape_html(word)
        safe_url = escape_html(url)
        # Replace only the full word with the hyperlink (HTML)
        answer = re.sub(
            rf"(?<!\w)({re.escape(word)})(?!\w)",
            f'<a href="{safe_url}">{safe_word}</a>',
            answer
        )
    return answer

async def generate_response(prompt: str, _, __=None) -> str:
    try:
        logger.info("Sending request to Groq API...")
        import groq
        client = groq.AsyncGroq(api_key=CURRENT_AI_API_KEY)
        chat_completion = await client.chat.completions.create(
            model=CURRENT_AI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800,
            top_p=0.95
        )
        answer = chat_completion.choices[0].message.content
        logger.info("Received response from Groq API")
        return answer.strip()
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}")
        raise RuntimeError(f"Failed to generate response: {str(e)}")

@rate_limit()
@log_command
async def ask_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    question = " ".join(context.args)
    if not question:
        await update.message.reply_text(
            "Please provide a question after the /ask command. For example:\n"
            "/ask Whose birthdays are in the month of April?"
        )
        return
    thinking_message = await update.message.reply_text("💣")
    # PATCH: Use search-then-summarize for context
    context_text = get_context_for_question(question, top_n=8)
    if not context_text.strip():
        await thinking_message.delete()
        await update.message.reply_text("No knowledge entries found to answer your question.")
        return
    try:
        await load_llm()
        prompt = build_prompt(question, context_text)
        relevant_entries = storage.search_entries(question, top_n=8)
        keywords = {entry["text"]: entry["link"] for entry in relevant_entries}
        answer = await generate_response(prompt, None, None)
        final_answer = add_hyperlinks(answer, keywords)
        output = f"{final_answer}"
        await thinking_message.delete()
        if len(output) > 4000:
            output = output[:3900] + "\n\n... (message truncated due to length)"
        try:
            await update.message.reply_html(
                clean_telegram_html(output),
                disable_web_page_preview=True
            )
        except Exception as e:
            logger.error(f"Error sending response: {str(e)}")
            await update.message.reply_text(
                "fool !! I'm 𝘯𝘰𝘵 𝘺𝘰𝘶𝘳 𝘴𝘦𝘳𝘷𝘢𝘯𝘵!"
            )
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        await thinking_message.delete()
        await update.message.reply_text("An error occurred while processing your question.")
                                        
@admin_only
async def clear_all_entries_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    category = None
    if context.args:
        category = " ".join(context.args)
    confirm_text = "Are you sure you want to clear "
    if category:
        confirm_text += f"all entries in category '{category}'?"
    else:
        confirm_text += "ALL entries?"
    keyboard = [
        [
            InlineKeyboardButton("Yes, Clear", callback_data=f"confirm_clear:{category or 'all'}"),
            InlineKeyboardButton("Cancel", callback_data="cancel_clear")
        ]
    ]
    await update.message.reply_text(confirm_text, reply_markup=InlineKeyboardMarkup(keyboard))

@admin_only
async def download_csv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Export the database to CSV
    entries = storage.get_entries()
    if not entries:
        await update.message.reply_text("No entries found in the database.")
        return
    
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".csv", encoding="utf-8") as tmp_file:
        fieldnames = ["text", "link", "category"]
        writer = csv.DictWriter(tmp_file, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow({
                "text": entry["text"],
                "link": entry["link"],
                "category": entry["category"]
            })
        tmp_path = tmp_file.name
    
    try:
        await update.message.reply_document(
            document=open(tmp_path, "rb"),
            filename="entries.csv",
            caption="Here's your complete knowledge base as a CSV file."
        )
    finally:
        # Clean up the temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass

@admin_only
async def request_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Request the user to upload a file (CSV or SQLite DB)."""
    categories = storage.get_categories()
    category_list = ", ".join(categories)
    
    keyboard = [
        [InlineKeyboardButton("CSV File", callback_data="upload:csv"),
         InlineKeyboardButton("SQLite DB", callback_data="upload:db")]
    ]
    
    await update.message.reply_text(
        "Please choose the type of file you want to upload:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def handle_upload_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle selection of upload file type."""
    query = update.callback_query
    await query.answer()
    
    data = query.data.split(":")[1]
    
    if data == "csv":
        categories = storage.get_categories()
        category_list = ", ".join(categories)
        
        await query.edit_message_text(
            "Please upload your CSV file as a reply to this message.\n\n"
            f"The file should have these columns: 'text', 'link', 'category'\n\n"
            f"Available categories: {category_list}\n"
        )
        context.user_data["awaiting_csv"] = True
        
    elif data == "db":
        await query.edit_message_text(
            "Please upload your SQLite database file (.db) as a reply to this message.\n\n"
            "This will replace the current database. All entries will be updated "
            "based on the uploaded file.\n\n"
            "⚠️ WARNING: This operation cannot be undone. Make sure to download a backup first."
        )
        context.user_data["awaiting_db"] = True
    
async def handle_upload_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle selection of upload file type."""
    query = update.callback_query
    await query.answer()
    
    data = query.data.split(":")[1]
    
    if data == "csv":
        categories = storage.get_categories()
        category_list = ", ".join(categories)
        
        await query.edit_message_text(
            "Please upload your CSV file as a reply to this message.\n\n"
            f"The file should have these columns: 'text', 'link', 'category'\n\n"
            f"Available categories: {category_list}\n"
        )
        context.user_data["awaiting_csv"] = True
        
    elif data == "db":
        await query.edit_message_text(
            "Please upload your SQLite database file (.db) as a reply to this message.\n\n"
            "This will replace the current database. All entries will be updated "
            "based on the uploaded file.\n\n"
            "⚠️ WARNING: This operation cannot be undone. Make sure to download a backup first."
        )
        context.user_data["awaiting_db"] = True

@admin_only
async def handle_csv_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message.reply_to_message and update.message.document:
        context.user_data["awaiting_csv"] = True
    if not context.user_data.get("awaiting_csv"):
        return
    context.user_data["awaiting_csv"] = False
    if not update.message.document:
        await update.message.reply_text("Please upload a CSV file.")
        return
    document = update.message.document
    file_name = document.file_name.lower() if document.file_name else "unnamed.file"
    if not file_name.endswith(".csv"):
        await update.message.reply_text("Please upload a file with .csv extension.")
        return
    
    uploaded_entries = []
    processing_status = await update.message.reply_text("⏳ Processing your uploaded file...")
    try:
        file = await context.bot.get_file(document.file_id)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            await file.download_to_drive(temp_file.name)
            file_path = temp_file.name
        file_content = ""
        rows_parsed = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    file_content = f.read()
            except Exception as e:
                await processing_status.edit_text(f"Error reading file: {str(e)}")
                if os.path.exists(file_path):
                    os.unlink(file_path)
                return
        if not any(separator in file_content for separator in [",", "\t", ";"]):
            await processing_status.edit_text(
                "The uploaded file doesn't appear to be in CSV format. Expected comma, tab, or semicolon delimiters."
            )
            if os.path.exists(file_path):
                os.unlink(file_path)
            return
        for delimiter in [",", "\t", ";"]:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter=delimiter)
                    if not reader.fieldnames:
                        continue
                    required_headers = ["text", "link"]
                    if all(header in reader.fieldnames for header in required_headers):
                        rows_parsed = list(reader)
                        await processing_status.edit_text(f"✅ Found CSV format with {delimiter} delimiter")
                        break
            except Exception as e:
                logger.info(f"Parsing with delimiter {delimiter} failed: {str(e)}")
                continue
        if not rows_parsed:
            await processing_status.edit_text(
                "Could not find required 'text' and 'link' columns in the CSV. "
                "Please check the file format and try again."
            )
            if os.path.exists(file_path):
                os.unlink(file_path)
            return
        # Process the successfully parsed rows
        for i, row in enumerate(rows_parsed, 1):
            try:
                text = row.get("text", "").strip()
                link = row.get("link", "").strip()
                if not text or not link:
                    logger.warning(f"Skipping row {i}: Missing required text or link field")
                    continue
                new_entry = {
                    "text": text,
                    "link": link,
                    "category": row.get("category", "General").strip() or "General"
                }
                uploaded_entries.append(new_entry)
            except Exception as row_error:
                logger.error(f"Error processing row {i}: {str(row_error)}")
        if os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except Exception as e:
                logger.error(f"Error cleaning up temp file: {str(e)}")
                
        if not uploaded_entries:
            await processing_status.edit_text("No valid entries found in the CSV file.")
            return
        await processing_status.delete()
        message = f"✅ Found {len(uploaded_entries)} valid entries in the CSV file. Do you want to:"
        keyboard = [
            [
                InlineKeyboardButton("Replace All", callback_data="csv:replace"),
                InlineKeyboardButton("Append", callback_data="csv:append"),
            ],
            [InlineKeyboardButton("Cancel", callback_data="csv:cancel")]
        ]
        context.user_data["uploaded_entries"] = uploaded_entries
        await update.message.reply_text(message, reply_markup=InlineKeyboardMarkup(keyboard))
    except Exception as e:
        logger.error(f"Error processing CSV upload: {str(e)}")
        try:
            await processing_status.delete()
        except:
            pass
        await update.message.reply_text(f"Error processing the uploaded file: {str(e)}")
        if os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except:
                pass

async def handle_csv_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    action = query.data.split(":", 1)[1]
    if action == "cancel":
        await query.edit_message_text("CSV import canceled.")
        return
    uploaded_entries = context.user_data.get("uploaded_entries", [])
    if not uploaded_entries:
        await query.edit_message_text("No entries to process.")
        return
    try:
        if action == "replace":
            # Clear all existing entries before adding new ones
            storage.clear_entries()
            success = True
            added_count = 0
            for entry in uploaded_entries:
                if storage.add_entry(entry["text"], entry["link"], entry["category"]):
                    added_count += 1
            message = f"✅ Successfully replaced all entries with {added_count} new entries."
        elif action == "append":
            # Add entries one by one
            added_count = 0
            skipped_count = 0
            for entry in uploaded_entries:
                if storage.add_entry(entry["text"], entry["link"], entry["category"]):
                    added_count += 1
                else:
                    skipped_count += 1
            message = f"✅ Added {added_count} new entries (skipped {skipped_count} duplicates)."
        await query.edit_message_text(message)
    except Exception as e:
        logger.error(f"Error handling CSV action: {str(e)}")
        await query.edit_message_text(f"Error: {str(e)}")

async def handle_csv_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    action = query.data.split(":", 1)[1]
    if action == "cancel":
        await query.edit_message_text("CSV import canceled.")
        return
    uploaded_entries = context.user_data.get("uploaded_entries", [])
    if not uploaded_entries:
        await query.edit_message_text("No entries to process.")
        return
    try:
        if action == "replace":
            success = write_entries(uploaded_entries)
            message = f"✅ Successfully replaced all entries with {len(uploaded_entries)} new entries." if success else "❌ Failed to update entries."
        elif action == "append":
            current_entries = read_entries()
            new_entries = []
            existing_count = 0
            for entry in uploaded_entries:
                is_duplicate = False
                for existing in current_entries:
                    if (existing["text"] == entry["text"] and 
                        existing["link"] == entry["link"]):
                        is_duplicate = True
                        existing_count += 1
                        break
                if not is_duplicate:
                    new_entries.append(entry)
            combined_entries = current_entries + new_entries
            success = write_entries(combined_entries)
            message = f"✅ Added {len(new_entries)} new entries (skipped {existing_count} duplicates)." if success else "❌ Failed to update entries."
        await query.edit_message_text(message)
    except Exception as e:
        logger.error(f"Error handling CSV action: {str(e)}")
        await query.edit_message_text(f"Error: {str(e)}")'''

# Modify your main function to use the status monitor
async def main():
    """Start the bot."""
    # Load environment variables
    load_dotenv()
    
    # Enhanced logging at startup
    logger.info("================ BOT STARTUP =================")
    logger.info(f"Starting bot at {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    # Initialize bot token and admin IDs
    bot_token = BOT_TOKEN
    logger.info(f"Bot token length: {len(bot_token) if bot_token else 'Token not found'}")
    
    # Validate bot token format
    if not bot_token or ":" not in bot_token:
        logger.error(f"Invalid bot token format. Token should contain ':' character. Please check your BOT_TOKEN.")
        raise ValueError("Invalid bot token format")
        
    # Test bot token with a simple getMe API call
    try:
        logger.info("Testing bot token with getMe API call...")
        response = requests.get(f"https://api.telegram.org/bot{bot_token}/getMe")
        if response.status_code != 200:
            error_data = response.json()
            logger.error(f"Bot token validation failed: {error_data}")
            raise ValueError(f"Invalid bot token: {error_data.get('description', 'Unknown error')}")
        bot_info = response.json().get('result', {})
        logger.info(f"Bot token validated successfully! Connected as @{bot_info.get('username')}")
    except Exception as e:
        logger.error(f"Error validating bot token: {str(e)}")
        raise ValueError(f"Failed to validate bot token: {str(e)}")
        
    admin_ids = [int(id.strip()) for id in os.getenv("ADMIN_USER_IDS", "").split(",") if id.strip()]
    logger.info(f"Admin IDs configured: {admin_ids}")
    
    # Initialize status monitor
    status_monitor = BotStatusMonitor(bot_token, admin_ids)
    # Schedule the daily backup of the entries CSV to the logs channel
    schedule_daily_db_backup(
        bot_token=bot_token,
        file_path="entries.db",
        channel_id=-1001925908750
    )
    
    # Send startup notification
    await status_monitor.send_startup_notification()
    
    try:
        # Initialize your application
        # Create application with detailed logging
        logger.info("Creating Application instance with the provided token...")
        try:
            application = Application.builder().token(bot_token).build()
            logger.info("Application instance created successfully")
        except Exception as e:
            logger.critical(f"Failed to create Application instance: {e}", exc_info=True)
            raise ValueError(f"Invalid bot configuration: {str(e)}")
        
        # Add handlers
        application.add_handler(CommandHandler("start", start))
        # ... other handlers ...
        # Standard command handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", start))
        application.add_handler(CommandHandler("setmodel", set_model_command))   # <--- PATCH: Add this line
        application.add_handler(CommandHandler("setapikey", set_apikey_command)) # <--- PATCH: Add this line
        application.add_handler(CommandHandler("list", list_entries))  # Now admin-only
        application.add_handler(CommandHandler("add", add_entry_command))  # Still admin-only
        application.add_handler(CommandHandler("ask", ask_question))  # Available to all users
        application.add_handler(CommandHandler("download", download_db))  # Admin-only db download
        application.add_handler(CommandHandler("download_csv", download_csv))  # Admin-only CSV export
        application.add_handler(CommandHandler("upload", request_upload))  # Admin-only upload selection
        application.add_handler(CommandHandler("clear", clear_all_entries_command))  # New admin-only command
        application.add_handler(CommandHandler("here", here_command))  # Available to all users
        application.add_handler(CommandHandler("logs", show_logs))  # New logs command
        application.add_handler(CommandHandler("s", show_entry_command))
        application.add_handler(CommandHandler("i", insert_entry_command))
        application.add_handler(CommandHandler("slowmode", slowmode_command))  # Admins only
        
        # File upload handlers
        application.add_handler(MessageHandler(
            filters.Document.FileExtension(".db") & filters.REPLY,
            handle_db_upload
        ))
        application.add_handler(MessageHandler(
            filters.Document.FileExtension(".csv") & filters.REPLY,
            handle_csv_upload
        ))
        
        # Enhanced callback query handlers
        application.add_handler(CallbackQueryHandler(handle_upload_selection, pattern=r"^upload:"))
        application.add_handler(CallbackQueryHandler(handle_csv_action, pattern=r"^csv:"))
        application.add_handler(CallbackQueryHandler(handle_pagination, pattern=r"^page:"))
        application.add_handler(CallbackQueryHandler(handle_pagination, pattern=r"^delete:"))
        application.add_handler(CallbackQueryHandler(handle_pagination, pattern=r"^cat:"))
        application.add_handler(CallbackQueryHandler(handle_pagination, pattern=r"^clear:"))
        application.add_handler(CallbackQueryHandler(handle_pagination, pattern=r"^confirm_clear:"))
        application.add_handler(CallbackQueryHandler(handle_pagination, pattern=r"^cancel_clear$"))
        application.add_handler(CallbackQueryHandler(handle_single_entry_delete, pattern=r"^sdelete:\d+$"))
        # Start health check server (if needed)
        # Note: We comment this out because the run_health_server function might not exist
        # health_thread = threading.Thread(target=run_health_server, daemon=True)
        # health_thread.start()
        
        # Schedule periodic health checks
        async def periodic_health_check():
            while True:
                await status_monitor.send_health_check()
                await asyncio.sleep(3600)  # Check every hour
        
        # Create the health check task
        health_check_task = asyncio.create_task(periodic_health_check())
        
        # Run the bot (only run once)
        logger.info("Starting bot polling...")
        try:
            # Configure telegram bot with more detailed logging
            # Set logging level for the application
            logging.getLogger('httpx').setLevel(logging.INFO)
            # Enable debug mode in telegram bot api
            application.bot._log = True
            # Detailed logging of telegram updates
            logger.info(f"Bot will listen for the following update types: {Update.ALL_TYPES}")
#            logger.info(f"Bot username: {application.bot.username}, Bot ID: {application.bot.id}")
            
            # Start polling with aggressive settings to ensure we get updates
            await application.run_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True,
                poll_interval=1.0,
                timeout=30
            )
        except Exception as e:
            logger.critical(f"Error in run_polling: {e}", exc_info=True)
            raise
    except Exception as e:
        # Send shutdown notification with error
        await status_monitor.send_shutdown_notification(f"Error: {str(e)}")
        raise
    finally:
        # Send shutdown notification
        await status_monitor.send_shutdown_notification()


if __name__ == "__main__":
    # Since we're using nest_asyncio, we can run both Flask and the Telegram bot
    import threading
    import asyncio
    
    def start_flask():
        try:
            # Try to use waitress for production deployment
            try:
                from waitress import serve
                logger.info("Starting Flask server with waitress on port 8081")
                serve(app, host="0.0.0.0", port=8081)
            except ImportError:
                # Fallback to Flask's built-in server
                logger.info("Waitress not available, using Flask's built-in server on port 8081")
                app.run(host="0.0.0.0", port=8081, debug=False)
        except Exception as e:
            logger.error(f"Failed to start Flask server: {e}")
    
    # Start Flask in a background thread
    flask_thread = threading.Thread(target=start_flask)
    flask_thread.daemon = True  # Thread will exit when main thread exits
    flask_thread.start()
    
    # Start the Telegram bot in the main thread
    logger.info("Starting Summarizer2 Telegram Bot")
    try:
        # With nest_asyncio.apply() already called, we can use asyncio.run safely
        logger.info("Initializing main event loop")
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by keyboard interrupt (Ctrl+C)")
    except Exception as e:
        logger.critical(f"Critical error in main process: {e}", exc_info=True)
        # Print stacktrace for easier debugging
        import traceback
        traceback.print_exc()
