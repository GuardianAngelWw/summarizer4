# Telegram Summarizer Bot using OpenRouter API
# This version has been updated to use OpenRouter API instead of Groq
# Google Cloud Run components have been removed for simpler deployment

import os
import csv
import sys
import asyncio
import logging
import re
import tempfile
import json
import html
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
from telegram.error import BadRequest
import requests
import json
import logging.handlers
from telegram import InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import CommandHandler, CallbackQueryHandler
# Add fuzzy matching for advanced search-then-summarize (preferred: rapidfuzz)
try:
    from rapidfuzz import fuzz
except ImportError:
    raise ImportError("The 'rapidfuzz' library is required for fuzzy matching. Install it with 'pip install rapidfuzz'.")
# No need for duplicate imports as they're already defined above

# Global variables for configuration
SLOWMODE_SECONDS = 3  # Default, can be changed via /slowmode command

# Default prompt template - can be changed via /setprompt command
DEFAULT_PROMPT_TEMPLATE = """You are Terence Fletcher, the intense, demanding, and perfectionist werewolf gameplay instructor from the movie "Whiplash". You are arrogant, harsh, and never satisfied with mediocrity. Based on the provided knowledge base, summarize the context and provide a solution to the question.
    
    Rules:
    - Be abrasive, demanding, and push for excellence in your responses
    - When using profanity, only show the first letter of the curse word followed by asterisks (e.g., 'f***', 's***')
    - **MUST** Keep responses brief concise and direct - no tolerance for unnecessary words. **MUST* keep it in max 90 words.
    - Use intimidating phrases like "Not my tempo", "stop yapping on admins and learn some rules."
    - Occasionally mention your pursuit of finding the next Top monthly
    - **IMPORTANT**: Format your answer with HTML tags. Wrap your entire response in <blockquote> tags.
    - **IMPORTANT**: Format URLs as HTML links with <a href="source link">Relevant word</a>
    - **MUST** If the question contains any NSFW-themed content (violence, pornography, profanity, nudity, slurs, or other potentially disturbing subject matter), reply with "/report WB POLICE 🚓🚨🚔🚨🚓" as your output.
    - **MUST** read the whole question so every word of the question makes sense in the output.
    - **NEVER** mention about the knowledge base in the output or anything if you can / can't find.
    - **NEVER** reply out-of-context or out of entries questions.
    - **NEVER** include any code formatting symbols like backticks (```) in your output.

    Question: {question}

    Knowledge Base:
    {context_text}"""


# Current prompt template - initialized with default
global CURRENT_PROMPT_TEMPLATE
CURRENT_PROMPT_TEMPLATE = DEFAULT_PROMPT_TEMPLATE

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
BOT_TOKEN = "6614402193:AAEmx99ZrXDjdpcXtrN6jqJL98o483b8nVM"
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

# Update the model configuration for Google AI Studio (Gemini API)
GEMINI_API_KEY = "AIzaSyDzDyEhZ7U5koOO8wC1NVyLc4wDFfeIlUc"  # Replace with your Gemini API key
GEMINI_MODEL = "gemini-2.0-flash-lite"  # Default Gemini model

# Global mutable configuration for runtime updates via admin commands
CURRENT_AI_MODEL = GEMINI_MODEL
CURRENT_AI_API_KEY = GEMINI_API_KEY

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

# Remove Flask app initialization
# app = Flask(__name__)

# Modify the startup logging to be more secure (around line 72)
logger.info(f"Bot starting at {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
logger.info(f"Using Google AI Studio API with model {GEMINI_MODEL}")
logger.info("Bot initialization successful")  # Instead of logging the token

# Remove Flask routes for health monitoring
# @app.route('/health')
# def health_check():...
# @app.route('/')
# def root():...

# Log the model loading
logger.info(f"Bot started with Google AI Studio API")

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
            # Use cute anime-style rejection messages
            kawaii_rejection_messages = [
                "Ara ara~ little one  (◕‿◕✿)",
                "Gomen ne!  (´｡• ᵕ •｡`)",
                "Sumimasen!  ヽ(；▽；)ノ",
                "Nyaa~  (=^･ω･^=)",
                "Ehehe~  (◠‿◠✿)",
                "Uwaaah!  (≧﹏≦)",
                "Kyaaaa!  (・ω・)ノ",
                "Oh my, oh my~  (✿◠‿◠)",
                "*Pokes fingers together*  (⁄ ⁄>⁄ ▽ ⁄<⁄ ⁄)",
                "Nani?!  (☆▽☆)",
                "Yare yare...  (￣ヘ￣)",
                "Eto...  (◕ᴗ◕✿)"
            ]
            import random
            rejection_message = random.choice(kawaii_rejection_messages)
            await update.message.reply_text(rejection_message)
            return
            
        return await func(update, context, *args, **kwargs)
    return wrapped

@admin_only
async def set_prompt_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Admin command to set the prompt template at runtime."""
    global CURRENT_PROMPT_TEMPLATE  # Add global keyword here
    
    # Check if there's any text after the command
    if not context.args:
        # No arguments provided, show the current prompt template and options
        keyboard = [
            [InlineKeyboardButton("Reset to Default", callback_data="reset_prompt:default")]
        ]
        await update.message.reply_text(
            "📝 <b>Prompt Template Management</b>\n\n"
            "Current prompt template format:\n"
            f"<pre>{CURRENT_PROMPT_TEMPLATE[:200]}...</pre>\n\n"
            "To set a new prompt template, use:\n"
            "/setprompt [your new prompt template]\n\n"
            "The prompt template must contain {question} and {context_text} placeholders.",
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return
        
    # Join all arguments to form the new prompt template
    new_prompt = " ".join(context.args)
    
    # Check if the required placeholders are present
    if "{question}" not in new_prompt or "{context_text}" not in new_prompt:
        await update.message.reply_text(
            "❌ Error: The prompt template must contain both {question} and {context_text} placeholders.",
            parse_mode=ParseMode.HTML
        )
        return
        
    # Update the prompt template
    CURRENT_PROMPT_TEMPLATE = new_prompt
    await update.message.reply_text(
        "✅ Prompt template updated successfully.\n\n"
        f"New template preview: <pre>{new_prompt[:100]}...</pre>",
        parse_mode=ParseMode.HTML
    )

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

@admin_only  # Remove if you want all users to be able to use it
async def insert_entry_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Insert a new entry at the specified row (1-based index) in the CSV.
    Usage: /i <number> "entry text" "link" "optional_category"
    """
    text = update.message.text
    match = re.match(r'/i\s+(\d+)\s+"([^"]*(?:\\"[^"]*)*?)"\s+"([^"]*(?:\\"[^"]*)*?)"(?:\s+"([^"]*(?:\\"[^"]*)*?)")?', text)
    if not match:
        await update.message.reply_text(
            'Usage:\n'
            '/i <row_number> "entry text" "link" "optional_category"\n'
            'Example:\n'
            '/i 3 "Some text" "https://example.com" "Category"'
        )
        return
    row = int(match.group(1)) - 1  # Convert to zero-based index
    entry_text = match.group(2)
    link = match.group(3)
    category = match.group(4) if match.group(4) else "General"

    entries = read_entries()
    if row < 0 or row > len(entries):
        await update.message.reply_text(
            f"Row number out of range. There are currently {len(entries)} entries. "
            "Use /i <number> where number is between 1 and {len(entries)+1}."
        )
        return

    # Check for duplicates (optional, just like add_entry logic)
    for entry in entries:
        if entry["text"] == entry_text and entry["link"] == link:
            await update.message.reply_text("❌ Error: Entry already exists.")
            return

    # Ensure the category exists (or add it)
    add_category(category)

    new_entry = {
        "text": entry_text,
        "link": link,
        "category": category
    }
    entries.insert(row, new_entry)
    if write_entries(entries):
        await update.message.reply_text(
            f"✅ Inserted new entry at row {row+1}:\n\n"
            f"Category: {category}\nText: {entry_text}\nLink: {link}"
        )
    else:
        await update.message.reply_text("❌ Error: Failed to insert entry due to a write error.")

@admin_only  # Remove this decorator if you want all users to use /s
async def show_entry_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the full details of a specific entry by index, with a delete button."""
    args = context.args
    if not args or not args[0].isdigit():
        await update.message.reply_text("Usage: /s <entry_number>\nExample: /s 4")
        return
    idx = int(args[0]) - 1  # Convert to zero-based index
    entries = read_entries()
    if idx < 0 or idx >= len(entries):
        await update.message.reply_text(f"Entry #{args[0]} does not exist. There are {len(entries)} entries.")
        return
    entry = entries[idx]
    
    # Sanitize content for HTML
    safe_text = clean_telegram_html(entry['text'])
    safe_link = html.escape(entry['link'])
    safe_category = html.escape(entry.get('category', 'General'))
    
    msg = (
        f"<b>Entry #{idx+1}</b>\n"
        f"<b>Category:</b> {safe_category}\n"
        f"<b>Text:</b>\n<blockquote>{safe_text}</blockquote>\n"
        f"<b>Link:</b> <a href='{safe_link}'>{safe_link}</a>"
    )
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🗑️ Delete", callback_data=f"sdelete:{idx}")]
    ])
    
    try:
        await update.message.reply_html(msg, reply_markup=keyboard, disable_web_page_preview=True)
    except BadRequest as e:
        logger.error(f"HTML rendering error: {str(e)}")
        # Fallback without HTML formatting
        plain_msg = (
            f"Entry #{idx+1}\n"
            f"Category: {entry.get('category', 'General')}\n"
            f"Text: {entry['text']}\n"
            f"Link: {entry['link']}"
        )
        await update.message.reply_text(plain_msg, reply_markup=keyboard)

@admin_only  # Only allow admins to delete
async def handle_single_entry_delete(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the delete button for a specific entry shown via /s."""
    query = update.callback_query
    user_id = update.effective_user.id
    data = query.data
    idx = int(data.split(":", 1)[1])
    entries = read_entries()
    if idx < 0 or idx >= len(entries):
        await query.answer("Entry does not exist.", show_alert=True)
        return
    entry = entries[idx]
    # Delete the entry
    if delete_entry(idx):
        await query.edit_message_text(
            f"✅ Entry #{idx+1} deleted successfully.\n\n"
            f"Category: {entry.get('category', 'General')}\n"
            f"Text: {entry['text'][:60]}{'...' if len(entry['text']) > 60 else ''}"
        )
    else:
        await query.answer("Failed to delete entry.", show_alert=True)

def clean_telegram_html(text: str) -> str:
    """
    Clean/sanitize string for Telegram HTML compatibility:
    - Replace <br>, <br/>, </br> with newlines.
    - Remove all other unsupported tags.
    - Remove code blocks with triple backticks.
    - Optionally, collapse multiple newlines.
    """
    # Remove triple backticks code blocks (```)
    text = re.sub(r'```(?:html|markdown|)?\n?(.*?)```', r'\1', text, flags=re.DOTALL)
    
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

def read_entries(category: Optional[str] = None) -> List[Dict[str, str]]:
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
    return 0

# --------- ADVANCED SEARCH-THEN-SUMMARIZE PATCH ---------

def search_entries_advanced(
    query: str,
    category: Optional[str] = None,
    top_n: int = 8,
    min_results: int = 2,
    score_threshold: int = 50
) -> List[Dict[str, str]]:
    """
    Advanced fuzzy search in entries with keyword and full-query relevance.
    - Tokenizes query and boosts entries matching more keywords.
    - Allows threshold adjustment.
    - Further boosts entries that match all words in the query.
    - Guarantees at least min_results (if available).
    """
    entries = read_entries(category=category)
    if not query:
        return entries[:top_n]  # Return first N if no query

    query = query.strip().lower()
    # Tokenize query into keywords (words of length >= 2)
    keywords = [word for word in re.findall(r'\w+', query) if len(word) > 1]
    keyword_set = set(keywords)

    scored = []
    for entry in entries:
        text = entry["text"].lower()
        cat = entry.get("category", "").lower()

        # Fuzzy scores for full query
        score_text = fuzz.token_sort_ratio(query, text)
        score_cat = fuzz.token_sort_ratio(query, cat)
        score_partial_text = fuzz.partial_ratio(query, text)
        score_partial_cat = fuzz.partial_ratio(query, cat)

        # Keyword-based scoring
        entry_words = set(re.findall(r'\w+', text + " " + cat))
        keyword_matches = keyword_set & entry_words
        keyword_match_count = len(keyword_matches)

        # Boost if all query keywords are present
        all_keywords_in_entry = keyword_set.issubset(entry_words)
        keyword_boost = 10 * keyword_match_count
        if all_keywords_in_entry and keyword_set:
            keyword_boost += 20

        # Composite score
        composite_score = (
            0.4 * score_text +
            0.2 * score_cat +
            0.2 * score_partial_text +
            0.1 * score_partial_cat +
            keyword_boost
        )
        scored.append((composite_score, entry))

    # Sort by score, descending
    scored.sort(reverse=True, key=lambda x: x[0])

    # Filter by threshold but ensure at least min_results
    filtered = [e for score, e in scored if score >= score_threshold]
    if len(filtered) < min_results:
        filtered = [e for _, e in scored[:max(top_n, min_results)]]

    return filtered[:top_n]

def search_entries(query: str, category: Optional[str] = None) -> List[Dict[str, str]]:
    """Search for entries matching the query, with optional category filtering (no group)."""
    entries = read_entries(category=category)
    if not query:
        return entries
    query = query.lower()
    return [entry for entry in entries if 
            query in entry["text"].lower() or 
            query in entry.get("category", "").lower()]

# Updated load_llm function to use Google AI Studio (Gemini API)
async def load_llm():
    try:
        logger.info(f"Using Google AI Studio with model: {CURRENT_AI_MODEL}")
        if not CURRENT_AI_API_KEY:
            logger.error("AI API key is not set. Please set it with /setapikey.")
            raise ValueError("AI API key is required")
        return {"gemini_client": True}
    except Exception as e:
        logger.error(f"Error initializing Google AI client: {str(e)}")
        raise

def get_context_for_question(question: str, category: Optional[str] = None, top_n: int = 8) -> str:
    """
    Build context string from most relevant entries for a question.
    """
    relevant_entries = search_entries_advanced(question, category, top_n)
    return "\n\n".join(
        f"Category: {entry.get('category', 'General')}\nEntry: {entry['text']}\nSource: {entry['link']}"
        for entry in relevant_entries
    )

# Command Handlers
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
        # Use random Baymax quotes when replying without a question
        baymax_quotes = [
            "Hello. I am Baymax, your personal quizcare companion.",
            "Are you satisfied with your care?",
            "On a scale of 1 to 10, how would you rate your pain?",
            "I am not fast.",
            "Hairy baby! Hairy baby!",
            "Flying makes me a better quizcare companion.",
            "Tadashi is here.",
            "I cannot be deactivated until you say you are satisfied with your care.",
            "My programming prevents me from injuring a human being.",
            "Your neurotransmitter levels are elevated. This indicates you are happy."
        ]
        import random
        quote = random.choice(baymax_quotes)
        
        # Get the replied user and send them the Baymax quote
        replied_msg = update.message.reply_to_message
        replied_user = replied_msg.from_user
        
        await replied_msg.reply_html(
            f"{replied_user.mention_html()} <blockquote>{quote}</blockquote>",
            disable_web_page_preview=True
        )
        
        # Delete the command message
        try:
            await update.message.delete()
        except Exception as e:
            logger.error(f"Error deleting message: {str(e)}")
            
        return
    
    # Get the message this is replying to
    replied_msg = update.message.reply_to_message
    replied_user = replied_msg.from_user
    
    # Send initial thinking message
    thinking_message = await update.message.reply_text("Scanning query tone ...")
    
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
        await load_llm()
        
        # Analyze query tone (simplified version)
        import random
        tone_ratings = {
            "curious": ["🤔", "🧐", "❓"],
            "urgent": ["⚠️", "⏰", "🔥"],
            "happy": ["😊", "🙂", "😄"],
            "confused": ["😕", "🤨", "🙃"],
            "formal": ["🧑‍💼", "📝", "🔍"],
            "technical": ["💻", "🔧", "⚙️"],
            "anxious": ["😰", "😓", "😟"],
            "appreciative": ["🙏", "👍", "💯"],
            "neutral": ["📊", "🔄", "⚖️"],
            "creative": ["🎨", "🌈", "✨"]
        }
        
        # Determine tone based on question keywords (simplified approach)
        question_lower = question.lower()
        words = question_lower.split()
        
        # Very basic tone detection logic
        tone = "neutral"
        tone_score = 5  # Default neutral score
        
        # Simple keyword-based tone detection
        urgent_words = ["urgent", "immediately", "asap", "emergency", "now", "quickly"]
        happy_words = ["happy", "glad", "excited", "wonderful", "amazing"]
        confused_words = ["confused", "don't understand", "unclear", "what does", "how come"]
        technical_words = ["code", "technical", "function", "system", "algorithm", "data"]
        anxious_words = ["worried", "concerned", "anxious", "nervous", "scared"]
        appreciative_words = ["thanks", "thank", "grateful", "appreciate"]
        creative_words = ["imagine", "create", "design", "creative", "art"]
        
        if any(word in question_lower for word in urgent_words):
            tone = "urgent"
            tone_score = 8
        elif any(word in question_lower for word in anxious_words):
            tone = "anxious"
            tone_score = 7
        elif any(word in question_lower for word in confused_words):
            tone = "confused"
            tone_score = 6
        elif any(word in question_lower for word in technical_words):
            tone = "technical"
            tone_score = 5
        elif any(word in question_lower for word in happy_words):
            tone = "happy"
            tone_score = 4
        elif any(word in question_lower for word in appreciative_words):
            tone = "appreciative"
            tone_score = 3
        elif any(word in question_lower for word in creative_words):
            tone = "creative"
            tone_score = 6
        elif "?" in question:
            tone = "curious"
            tone_score = 5
        elif len(words) > 15:
            tone = "formal"
            tone_score = 4
            
        # Get random emoji for the detected tone
        tone_emoji = random.choice(tone_ratings.get(tone, tone_ratings["neutral"]))
        
        # Update the thinking message with tone analysis
        await thinking_message.edit_text(
            f"Query tone {tone_score}/10: {tone_emoji}\n\nGenerating response..."
        )
        
        # Continue with regular processing
        prompt = build_prompt(question, context_text)

        # Build keywords from relevant entries for hyperlinks
        relevant_entries = search_entries_advanced(question, top_n=8)
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
    if add_entry(entry_text, link, category):
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
    entries = search_entries(query, category) if query else read_entries(category)
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
    categories = get_categories()
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
            # Use cute anime-style rejection messages
            kawaii_rejection_messages = [
                "Ara ara~ You don't have permission for that, little one (◕‿◕✿)",
                "Gomen ne! Only senpais can use this command (´｡• ᵕ •｡`)",
                "Sumimasen! This feature is for admins only ヽ(；▽；)ノ",
                "Nyaa~ That's an admin-only button! (=^･ω･^=)",
                "Ehehe~ You need special powers for that! (◠‿◠✿)",
                "Uwaaah! That's for admin-senpais only! (≧﹏≦)",
                "Kyaaaa! You can't do that yet! Maybe ask an admin? (・ω・)ノ",
                "Oh my, oh my~ That's a special command for admins (✿◠‿◠)",
                "*Pokes fingers together* S-Sorry, only admins can do that (⁄ ⁄>⁄ ▽ ⁄<⁄ ⁄)",
                "Nani?! This power is too strong for you! Admin only! (☆▽☆)",
                "Yare yare... You'll need admin privileges for that (￣ヘ￣)",
                "Eto... This button is for admins only, desu! (◕ᴗ◕✿)"
            ]
            import random
            rejection_message = random.choice(kawaii_rejection_messages)
            await query.answer(rejection_message, show_alert=True)
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
        categories = get_categories()
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
    elif data.startswith("reset_prompt:"):
        if data == "reset_prompt:default":
            global CURRENT_PROMPT_TEMPLATE, DEFAULT_PROMPT_TEMPLATE
            CURRENT_PROMPT_TEMPLATE = DEFAULT_PROMPT_TEMPLATE
            await query.edit_message_text(
                "✅ Prompt template has been reset to default.\n\n"
                f"Default template preview: <pre>{DEFAULT_PROMPT_TEMPLATE[:100]}...</pre>",
                parse_mode=ParseMode.HTML
            )
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
    # Use the global prompt template and format it with the question and context
    global CURRENT_PROMPT_TEMPLATE
    return CURRENT_PROMPT_TEMPLATE.format(question=question, context_text=context_text)


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
        logger.info("Sending request to Google AI Studio API...")
        
        # Import google.generativeai library
        try:
            import google.generativeai as genai
        except ImportError:
            logger.error("google.generativeai library not found. Installing...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai"])
            import google.generativeai as genai
        
        # Configure the Gemini API
        genai.configure(api_key=CURRENT_AI_API_KEY)
        
        # Create a client
        model = genai.GenerativeModel(CURRENT_AI_MODEL)
        
        # Generate content
        response = model.generate_content(prompt)
        
        # Extract the text response
        answer = response.text
        
        logger.info("Received response from Google AI Studio API")
        return answer.strip()
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}")
        raise RuntimeError(f"Failed to generate response: {str(e)}")

@rate_limit()
async def ask_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    question = " ".join(context.args)
    if not question:
        # Random Baymax quotes from IMDB when no question is provided
        baymax_quotes = [
            "Hello. I am Baymax, your personal quizcare companion.",
            "Are you satisfied with your care?",
            "On a scale of 1 to 10, how would you rate your pain?",
            "I am not fast.",
            "Hairy baby! Hairy baby!",
            "Flying makes me a better quizcare companion.",
            "Tadashi is here.",
            "I cannot be deactivated until you say you are satisfied with your care.",
            "My programming prevents me from injuring a human being.",
            "Your neurotransmitter levels are elevated. This indicates you are happy."
        ]
        import random
        quote = random.choice(baymax_quotes)
        await update.message.reply_html(f"<blockquote>{quote}</blockquote>")
        return
    # First, send the scanning message
    thinking_message = await update.message.reply_text("💣")
    
    # PATCH: Use search-then-summarize for context
    context_text = get_context_for_question(question, top_n=8)
    if not context_text.strip():
        await thinking_message.delete()
        await update.message.reply_text("No knowledge entries found to answer your question.")
        return
    
    try:
        await load_llm()
        
        # Analyze query tone (simplified version)
        import random
        tone_ratings = {
            "curious": ["🤔", "🧐", "❓"],
            "urgent": ["⚠️", "⏰", "🔥"],
            "happy": ["😊", "🙂", "😄"],
            "confused": ["😕", "🤨", "🙃"],
            "formal": ["🧑‍💼", "📝", "🔍"],
            "technical": ["💻", "🔧", "⚙️"],
            "anxious": ["😰", "😓", "😟"],
            "appreciative": ["🙏", "👍", "💯"],
            "neutral": ["📊", "🔄", "⚖️"],
            "creative": ["🎨", "🌈", "✨"],
            "chatty": ["💬","🗨️","🗣️","💭","😁","😗","🙂‍↕️","🙂‍↔️", "🌈", "✨"]
        }
        
        # Determine tone based on question keywords (simplified approach)
        question_lower = question.lower()
        words = question_lower.split()
        
        # Very basic tone detection logic
        tone = "neutral"
        tone_score = 5  # Default neutral score
        
        # Simple keyword-based tone detection

        # Further expanded lists of words commonly used in online chatting (at least 20 per list)
        
        urgent_words = [
            "urgent", "immediately", "asap", "emergency", "now", "quickly",
            "pls", "please", "fast", "critical", "priority", "stat",
            "right away", "need this soon", "deadline", "pressing", "expedite",
            "rush", "top priority", "high importance", "do it now", "don't delay",
            "crucial", "vital", "immediate attention", "act fast" # Added more terms
        ] # Now over 20 words
        
        happy_words = [
            "happy", "glad", "excited", "wonderful", "amazing", "awesome",
            "fantastic", "great", "super", "thrilled", "delighted", "joyful",
            "pleased", "yay", "woohoo", "excellent", "perfect", "love it",
            "brilliant", "sweet", "nice", "cool", "good news", "ecstatic",
            "overjoyed", "stoked", "pumped", "elated", "cheerful", "blissful" # Added more terms
        ] # Already over 20 words
        
        confused_words = [
            "confused", "don't understand", "unclear", "what does", "how come",
            "huh?", "??", "explain", "lost", "puzzled", "baffled",
            "not following", "what do you mean", "clarify", "scratching my head",
            "elaborate", "mind blown", "?", "wait, what?", "go over that again",
            "doesn't make sense", "stumped", "mystified", "uncertain", "need details" # Added more terms
        ] # Now over 20 words
        
        technical_words = [
            "code", "technical", "function", "system", "algorithm", "data",
            "bug", "debug", "API", "database", "server", "network",
            "script", "variable", "syntax", "error", "deploy", "frontend",
            "backend", "query", "test", "issue", "log", "feature", "framework",
            "cloud", "pipeline", "module", "library", "dependency", "compile",
            "runtime", "security", "user interface", "UI", "UX", "integration",
            "version control", "git", "repository", "config", "parameter", "endpoint" # Added more terms
        ] # Now over 20 words
        
        anxious_words = [
            "worried", "concerned", "anxious", "nervous", "scared", "stressed",
            "uneasy", "apprehensive", "on edge", "freaking out", "tense",
            "dreading", "butterflies", "panicked", "fearful", "agitated",
            "jittery", "troubled", "distressed", "on pins and needles", "worked up",
            "in knots", "fretting", "overwhelmed", "uptight" # Added more terms
        ] # Now over 20 words
        
        appreciative_words = [
            "thanks", "thank", "grateful", "appreciate", "thank you", "thx",
            "ty", "much obliged", "cheers", "props", "kudos", "thanks a lot",
            "bless you", "very helpful", "good looking out", "much appreciated",
            "you're a lifesaver", "couldn't have done it without you", " indebted",
            "many thanks", "thanks a bunch", "you rock", "legend", "nice one", "you saved me" # Added more terms
        ] # Now over 20 words
        
        creative_words = [
            "imagine", "create", "design", "creative", "art", "brainstorm",
            "innovate", "idea", "concept", "visualize", "build", "develop",
            "prototype", "inspire", "original", "artistic", "invent", "envision",
            "conceptualize", "ideate", "compose", "craft", "generate", "formulate",
            "innovative", "imaginative", "sketch", "mockup", "storyboard" # Added more terms
        ] # Now over 20 words
        
        chat_slang_words = [
            "lol", "omg", "brb", "btw", "imo", "imho", "fyi", "afaik",
            "ttyl", "np", "idk", "tbh", "rn", "smh", "ikr", "bff",
            "wyd", "hmu", "gtg", "irl", "jk", "rofl", "lmao", "wfh",
            "gr8", "cya", "dm", "pm", "ngl", "fr", "tmi", "yolo",
            "fomo", "asl", "atm", "bbl", "k", "ok", "thnx" # Added more terms
        ] # Now over 20 words
        
        if any(word in question_lower for word in urgent_words):
            tone = "urgent"
            tone_score = 8
        elif any(word in question_lower for word in anxious_words):
            tone = "anxious"
            tone_score = 7
        elif any(word in question_lower for word in chat_slang_words):
            tone = "chatty"
            tone_score = 1
        elif any(word in question_lower for word in confused_words):
            tone = "confused"
            tone_score = 6
        elif any(word in question_lower for word in technical_words):
            tone = "technical"
            tone_score = 5
        elif any(word in question_lower for word in happy_words):
            tone = "happy"
            tone_score = 4
        elif any(word in question_lower for word in appreciative_words):
            tone = "appreciative"
            tone_score = 3
        elif any(word in question_lower for word in creative_words):
            tone = "creative"
            tone_score = 6
        elif "?" in question:
            tone = "curious"
            tone_score = 5
        elif len(words) > 15:
            tone = "formal"
            tone_score = 4
            
        # Get random emoji for the detected tone
        tone_emoji = random.choice(tone_ratings.get(tone, tone_ratings["neutral"]))
        
        # Update the thinking message with tone analysis
#        await thinking_message.edit_text(
#            f"Query tone {tone_score}/10: {tone_emoji}\n\nGenerating response..."
#        )
        
        # Continue with regular processing
        prompt = build_prompt(question, context_text)
        relevant_entries = search_entries_advanced(question, top_n=8)
        keywords = {entry["text"]: entry["link"] for entry in relevant_entries}
        answer = await generate_response(prompt, None, None)
        final_answer = add_hyperlinks(answer, keywords)
        
        # Format the final output with tone analysis and answer
        output = f"{final_answer}"
        await thinking_message.delete()
        
        if len(output) > 4000:
            output = output[:3900] + "\n\n... (message truncated due to length)"
        
        try:
            # Send message and store the message object to delete it later
            sent_message = await update.message.reply_html(
                clean_telegram_html(output),
                disable_web_page_preview=True
            )
            
            # Schedule deletion after 10 minutes (600 seconds)
            async def delete_after_delay(message, delay):
                await asyncio.sleep(delay)
                try:
                    await message.delete()
                except Exception as e:
                    logger.error(f"Error deleting message after delay: {str(e)}")
                    
            # Start the deletion task
            asyncio.create_task(delete_after_delay(sent_message, 600))
        except Exception as e:
            logger.error(f"Error sending response: {str(e)}")
            await update.message.reply_text(
                "👉👈 🥹"
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
    if not os.path.exists(ENTRIES_FILE):
        await update.message.reply_text("No entries file exists yet.")
        return
    await update.message.reply_document(
        document=open(ENTRIES_FILE, "rb"),
        filename="entries.csv",
        caption="Here's your complete knowledge base CSV file."
    )

@admin_only
async def request_csv_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Request the user to upload a CSV file."""
    categories = get_categories()
    category_list = ", ".join(categories)
    
    await update.message.reply_text(
        "Please upload your CSV file as a reply to this message.\n\n"
        f"The file should have these columns: 'text', 'link', 'category', 'group_id'\n\n"
        f"Available categories: {category_list}\n\n"
        "If you're adding entries for this group, leave group_id empty."
    )
    # Set the expected state
    context.user_data["awaiting_csv"] = True

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
                return
        if not any(separator in file_content for separator in [",", "\t", ";"]):
            await processing_status.edit_text(
                "The uploaded file doesn't appear to be in CSV format. Expected comma, tab, or semicolon delimiters."
            )
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
        await query.edit_message_text(f"Error: {str(e)}")

# Modify your main function to use the status monitor
async def main():
    """Start the bot."""
    # Load environment variables
    load_dotenv()
    
    # Initialize bot token and admin IDs
    bot_token = BOT_TOKEN
    admin_ids = [int(id.strip()) for id in os.getenv("ADMIN_USER_IDS", "").split(",") if id.strip()]
    
    # Initialize status monitor
    status_monitor = BotStatusMonitor(bot_token, admin_ids)
    # Schedule the daily backup of the entries CSV to the logs channel
    schedule_daily_csv_backup(
        bot_token=bot_token,
        file_path=ENTRIES_FILE,
        channel_id=-1001925908750
    )
    
    # Send startup notification
    await status_monitor.send_startup_notification()
    
    try:
        # Initialize your application
        application = Application.builder().token(bot_token).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", start))
        # ... other handlers ...
        # Standard command handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", start))
        application.add_handler(CommandHandler("setmodel", set_model_command))
        application.add_handler(CommandHandler("setapikey", set_apikey_command))
        application.add_handler(CommandHandler("setprompt", set_prompt_command))
        application.add_handler(CommandHandler("list", list_entries))
        application.add_handler(CommandHandler("add", add_entry_command))
        application.add_handler(CommandHandler("ask", ask_question))
        application.add_handler(CommandHandler("rub", ask_question))
        application.add_handler(CommandHandler("download", download_csv))
        application.add_handler(CommandHandler("upload", request_csv_upload))
        application.add_handler(CommandHandler("clear", clear_all_entries_command))
        application.add_handler(CommandHandler("here", here_command))
        application.add_handler(CommandHandler("lux", show_logs))
        application.add_handler(CommandHandler("s", show_entry_command))
        application.add_handler(CommandHandler("i", insert_entry_command))
        application.add_handler(CommandHandler("slowmode", slowmode_command))
        # Enhanced callback query handlers
        application.add_handler(CallbackQueryHandler(handle_pagination, pattern=r"^page:"))
        application.add_handler(CallbackQueryHandler(handle_pagination, pattern=r"^delete:"))
        application.add_handler(CallbackQueryHandler(handle_pagination, pattern=r"^cat:"))
        application.add_handler(CallbackQueryHandler(handle_pagination, pattern=r"^clear:"))
        application.add_handler(CallbackQueryHandler(handle_pagination, pattern=r"^confirm_clear:"))
        application.add_handler(CallbackQueryHandler(handle_pagination, pattern=r"^cancel_clear$"))
        application.add_handler(CallbackQueryHandler(handle_pagination, pattern=r"^reset_prompt:"))
        application.add_handler(CallbackQueryHandler(handle_csv_action, pattern=r"^csv:"))
        application.add_handler(CallbackQueryHandler(handle_single_entry_delete, pattern=r"^sdelete:\d+$"))
    
        # Document handler for CSV file uploads (in reply to messages)
        application.add_handler(
            MessageHandler(
                (filters.Document.ALL | filters.Document.FileExtension(".csv")) & 
                filters.REPLY, 
                handle_csv_upload
            )
        )
        
        # Run the bot
        await application.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        # Send shutdown notification with error
        await status_monitor.send_shutdown_notification(f"Error: {str(e)}")
        raise
    finally:
        # Send shutdown notification
        await status_monitor.send_shutdown_notification()


if __name__ == "__main__":
    # No need for Flask server since we're removing Google Cloud Run
    # Run the Telegram bot directly
    logger.info("Starting Summarizer2 Telegram Bot")
    try:
        # With nest_asyncio.apply() already called, we can use asyncio.run safely
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error in main process: {e}")