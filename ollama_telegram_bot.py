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
# Groq API client will be imported as needed

# No need for duplicate imports as they're already defined above

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
BOT_TOKEN = "6614402193:AAGpKtzefMx23B9zk7LLt1JdLVuf9rJM-Pw"
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
ADMIN_USER_IDS = "5980915474"
ADMIN_USERS = [int(uid.strip()) for uid in ADMIN_USER_IDS.split(",") if uid.strip()]
CURRENT_DATE = "2025-04-27 09:19:30"  # Updated current UTC time
CURRENT_USER = "GuardianAngelWw"      # Updated current user
ENTRIES_FILE = "entries.csv"
CATEGORIES_FILE = "categories.json"
CSV_HEADERS = ["text", "link", "category", "group_id"]  # Added category and group_id fields

# Update the model configuration for Groq API
TOGETHER_API_KEY = os.getenv("GROQ_API_KEY", "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")
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # Using Groq compatible model

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
        return user_id in ADMIN_USERS or chat_member.status in ["administrator", "creator"]
    except Exception as e:
        logger.error(f"Error checking admin status: {str(e)}")
        return False

async def send_csv_to_logs_channel(bot_token: str, file_path: str, channel_id: int):
    """Send the CSV file to the specified Telegram channel."""
    try:
        app = Application.builder().token(bot_token).build()
        await app.bot.send_document(
            chat_id=channel_id,
            document=open(file_path, "rb"),
            filename=os.path.basename(file_path),
            caption="📦 Daily backup: Current entries.csv file."
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

# Updated load_llm function to use Groq API
async def load_llm():
    try:
        logger.info(f"Using Groq API with model: {GROQ_MODEL}")
        
        if not TOGETHER_API_KEY:  # Still using the TOGETHER_API_KEY variable name for now
            logger.error("GROQ_API_KEY is not set. Please set it in .env file or environment variables.")
            raise ValueError("GROQ_API_KEY is required")
        
        # Return a basic structure to confirm Groq setup
        return {"groq_client": True}
    except Exception as e:
        logger.error(f"Error initializing Groq client: {str(e)}")
        raise

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
    
    if is_user_admin:
        admin_text = (
            "/list - List knowledge entries (admin only)\n"
            "/add \"entry text\" \"message_link\" \"category\" - Add a new entry\n"
            "/download - Download the current CSV file\n"
            "/upload - Upload a CSV file\n"
            "/clear - Clear all entries or entries in a specific category\n"
        )
        help_text += admin_text
    
    help_text += "\nUse categories to organize your knowledge entries."
    await update.message.reply_html(help_text)
    
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
        # Load LLM components
        await thinking_message.edit_text("🤔")
        
        # Just validate API key
        await load_llm()
        
        await thinking_message.edit_text("⚡")
        
        # Create context from entries with categories
        context_entries = []
        for entry in entries:
            category = entry.get("category", "General")
            entry_text = f"Category: {category}\nEntry: {entry['text']}\nSource: {entry['link']}"
            context_entries.append(entry_text)
        
        context_text = "\n\n".join(context_entries)
        
        prompt = build_prompt(question, context_text)
        
        # Generate response
        answer = await generate_response(prompt, None, None)
        
        # Add hyperlinks to the answer
        keywords = {entry["text"]: entry["link"] for entry in entries}
        final_answer = add_hyperlinks(answer, keywords)
        
        # Send response to the replied user
        await thinking_message.delete()
        # Check if response is too long
        if len(final_answer) > 4000:  # Telegram has ~4096 char limit
            final_answer = final_answer[:3900] + "\n\n... (message truncated due to length)"
        # Inside the here_command function:
        try:
            await replied_msg.reply_text(
                f"{replied_user.mention_html()} 👇 {final_answer}",
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True  # Disable Telegram link previews and web previews
            )
        except Exception as e:
            logger.error(f"Error sending response: {str(e)}")
            # Send a simplified fallback message
            await replied_msg.reply_text(
                f"{replied_user.mention_html()}, I found an answer to your question, but had trouble formatting it.",
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
    """Add a new entry with optional category and group specificity."""
    # Check command format
    text = update.message.text[5:].strip()  # Remove "/add "
    
    # Extract text, link, and optional category using regex to handle quoted arguments
    # Handle quoted arguments pattern with proper escaping for nested quotes
    match = re.match(r'"([^"]*(?:\\"[^"]*)*?)"\s+"([^"]*(?:\\"[^"]*)*?)"(?:\s+"([^"]*(?:\\"[^"]*)*?)")?', text)
    
    if not match:
        await update.message.reply_text(
            "Please use the format: /add \"entry text\" \"message_link\" \"optional_category\""
        )
        return
    
    # Extract the matched groups
    match_groups = match.groups()
    entry_text = match_groups[0]
    link = match_groups[1]
    category = match_groups[2] if len(match_groups) > 2 and match_groups[2] else "General"
    
    # Get group ID if in a group
    group_id = None
    if update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP]:
        group_id = update.effective_chat.id
    
    # Add the entry
    if add_entry(entry_text, link, category, group_id):
        await update.message.reply_text(f"✅ Added new entry:\n\nCategory: {category}\nText: {entry_text}\nLink: {link}")
    else:
        await update.message.reply_text("❌ Error: Entry already exists or could not be added.")

@admin_only
async def list_entries(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List all entries with pagination, category filtering, and group specificity."""
    args = context.args if context.args else []
    
    # Parse arguments - format: ["query", "category=value"]
    query = ""
    category = None
    
    for arg in args:
        if arg.startswith("category="):
            category = arg.split("=")[1] if len(arg.split("=")) > 1 else None
        else:
            query = arg
    
    page = int(context.user_data.get('page', 0))
    
    # Determine group ID for group-specific entries
    group_id = None
    if update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP]:
        group_id = update.effective_chat.id
    
    # Get entries, filtered by search query, category, and group if provided
    entries = search_entries(query, group_id, category) if query else read_entries(group_id, category)
    
    # Calculate pagination
    total_pages = (len(entries) + ENTRIES_PER_PAGE - 1) // ENTRIES_PER_PAGE
    start_idx = page * ENTRIES_PER_PAGE
    end_idx = min(start_idx + ENTRIES_PER_PAGE, len(entries))
    
    # Generate message text
    if not entries:
        message = "No entries found."
        if category:
            message += f" in category '{category}'"
        if query:
            message += f" matching '{query}'"
        await update.message.reply_text(message)
        return
    
    # Build header message
    message = f"📚 Entries {start_idx+1}-{end_idx} of {len(entries)}"
    if category:
        message += f" in category '{category}'"
    if query:
        message += f" matching '{query}'"
    message += ":\n\n"
    
    # Add entries to message
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
    
    # Send message
    await update.message.reply_text(message, reply_markup=reply_markup)

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
    
    # Handle category selection
    if data.startswith("cat:"):
        category = data.split(":")[1]
        context.user_data['page'] = 0  # Reset page when changing category
        context.user_data['category'] = category
        
        # Re-trigger list with the selected category
        fake_update = Update(update.update_id, message=update.effective_message)
        await list_entries(fake_update, context)
        return
        
    # Handle page navigation
    if data.startswith("page:"):
        parts = data.split(":")
        page = int(parts[1])
        category = parts[2] if len(parts) > 2 and parts[2] else None
        
        context.user_data['page'] = page
        
        # Determine group ID for group-specific entries
        group_id = None
        if update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP]:
            group_id = update.effective_chat.id
        
        # Get entries with filtering
        entries = read_entries(group_id, category)
        
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

    # Rest of the function remains the same...
    elif data.startswith("delete:"):
        index = int(data.split(":")[1])
        if delete_entry(index):
            await query.edit_message_text(f"✅ Entry #{index+1} deleted successfully.")
        else:
            await query.edit_message_text(f"❌ Failed to delete entry #{index+1}.")
            
    # Handle clear all entries
    elif data.startswith("clear:"):
        category_filter = data.split(":")[1]
        category = None if category_filter == 'all' else category_filter
        
        # Determine group ID for group-specific entries
        group_id = None
        if update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP]:
            group_id = update.effective_chat.id
        
        # Confirm before clearing
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
        
    # Handle confirm clear action
    elif data.startswith("confirm_clear:"):
        category_filter = data.split(":")[1]
        category = None if category_filter == 'all' else category_filter
        
        # Determine group ID for group-specific entries
        group_id = None
        if update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP]:
            group_id = update.effective_chat.id
            
        count = clear_all_entries(group_id, category)
        if count > 0:
            await query.edit_message_text(f"✅ Successfully cleared {count} entries.")
        else:
            await query.edit_message_text("❌ No entries were cleared or an error occurred.")
            
    # Handle cancel clear action
    elif data == "cancel_clear":
        await query.edit_message_text("Operation cancelled.")

# Helper functions for ask_question
def build_prompt(question: str, context_text: str) -> str:
    return f"""You are an AI assistant with relatively high egoistic nature. Based on the provided knowledge base, summarize the context and provide a solution to the question without repeating it in the response:
    - While answering, ensure the answer is concise, understandable, and easy to follow and in very short brief.
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
    Replace keywords with Telegram markdown links in the answer.

    :param answer: The generated answer text.
    :param keywords: A dictionary of keywords and their corresponding URLs.
    :return: Updated answer with hyperlinks.
    """
    for word, url in keywords.items():
        # Replace only the full word or part of word with the hyperlink
        answer = re.sub(
            rf"(?<!\w)({re.escape(word)})(?!\w)",  # Match word boundaries to replace only intended parts
            f"[\\1]({url})",  # Telegram markdown format
            answer
        )
    return answer

async def generate_response(prompt: str, _, __=None) -> str:
    try:
        logger.info("Sending request to Groq API...")
        
        # Import groq only when needed
        import groq
        
        # Initialize Groq client
        client = groq.AsyncGroq(api_key=TOGETHER_API_KEY)  # Using the existing variable name but it now contains Groq API key
        
        # Use chat completions with proper formatting
        chat_completion = await client.chat.completions.create(
            model=GROQ_MODEL,  # Use the configured Groq model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800,  # Increased token count but still within safe limits
            top_p=0.95
        )
        
        # Extract the response
        answer = chat_completion.choices[0].message.content
        
        logger.info("Received response from Groq API")
        return answer.strip()
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}")
        raise RuntimeError(f"Failed to generate response: {str(e)}")

async def ask_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle the /ask command to respond to user questions.
    """
    question = " ".join(context.args)
    if not question:
        await update.message.reply_text(
            "Please provide a question after the /ask command. For example:\n"
            "/ask Whose birthdays are in the month of April?"
        )
        return

    # Send initial thinking message
    thinking_message = await update.message.reply_text("💣")

    # Load knowledge base
    group_id = update.effective_chat.id if update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP] else None
    entries = read_entries(group_id=group_id)
    if not entries:
        await thinking_message.delete()
        await update.message.reply_text("No knowledge entries found to answer your question.")
        return

    # Build context and prompt
    context_entries = "\n\n".join(
        f"Category: {entry.get('category', 'General')}\nEntry: {entry['text']}\nSource: {entry['link']}" for entry in entries
    )
    prompt = build_prompt(question, context_entries)

    # Generate response
    try:
        await load_llm()  # Just validate API key
        answer = await generate_response(prompt, None, None)

        # Add hyperlinks
        keywords = {entry["text"]: entry["link"] for entry in entries}
        final_answer = add_hyperlinks(answer, keywords)

        # Format the final answer in Markdown
        output = f"{final_answer}"

        # Send final response
        await thinking_message.delete()
        # Check message length to avoid Telegram error
        if len(output) > 4000:  # Telegram has ~4096 char limit
            output = output[:3900] + "\n\n... (message truncated due to length)"
        # Inside the ask_question function:
        try:
            await update.message.reply_html(
                output,
                disable_web_page_preview=True  # Disable Telegram link previews and web previews
            )
        except Exception as e:
            logger.error(f"Error sending response: {str(e)}")
            # Send a simplified fallback message
            await update.message.reply_text(
                "I found an answer to your question, but had trouble formatting it. Please try asking in a different way."
            )
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        await thinking_message.delete()
        await update.message.reply_text("An error occurred while processing your question.")
                                        
@admin_only
async def clear_all_entries_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command to clear all entries or by category."""
    # Parse arguments
    category = None
    if context.args:
        category = " ".join(context.args)
    
    # Determine group ID for group-specific entries
    group_id = None
    if update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP]:
        group_id = update.effective_chat.id
    
    # Confirm before clearing
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
    """Download the entries CSV file."""
    if not os.path.exists(ENTRIES_FILE):
        await update.message.reply_text("No entries file exists yet.")
        return
    
    # Get entries for the current group if in a group chat
    group_id = None
    if update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP]:
        group_id = update.effective_chat.id
        entries = read_entries(group_id)
        
        # Create a temporary file with only the group's entries
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            try:
                with open(temp_file.name, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                    writer.writeheader()
                    writer.writerows(entries)
                
                # Send the filtered CSV
                await update.message.reply_document(
                    document=open(temp_file.name, "rb"),
                    filename=f"entries_{update.effective_chat.title or 'group'}.csv",
                    caption=f"Here's your knowledge base CSV file for this group. ({len(entries)} entries)"
                )
            finally:
                # Clean up temp file
                os.unlink(temp_file.name)
    else:
        # For private chats, send the full CSV
        await update.message.reply_document(
            document=open(ENTRIES_FILE, "rb"),
            filename="entries.csv",
            caption=f"Here's your complete knowledge base CSV file."
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
    """Handle the uploaded CSV file."""
    # Always try to process CSV files when they're uploaded in reply
    if update.message.reply_to_message and update.message.document:
        # Set the state to true to ensure processing continues
        context.user_data["awaiting_csv"] = True
    
    if not context.user_data.get("awaiting_csv"):
        return
    
    # Reset state
    context.user_data["awaiting_csv"] = False
    
    if not update.message.document:
        await update.message.reply_text("Please upload a CSV file.")
        return
    
    document = update.message.document
    file_name = document.file_name.lower() if document.file_name else "unnamed.file"
    
    # More permissive file checking - accept any file and try to process as CSV
    uploaded_entries = []
    processing_status = await update.message.reply_text("⏳ Processing your uploaded file...")
    
    try:
        # Download the file
        file = await context.bot.get_file(document.file_id)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            await file.download_to_drive(temp_file.name)
            file_path = temp_file.name
        
        # Try multiple parsing approaches
        file_content = ""
        rows_parsed = []
        
        # First read the raw content to examine
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
        except UnicodeDecodeError:
            # Try different encodings if UTF-8 fails
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    file_content = f.read()
            except Exception as e:
                await processing_status.edit_text(f"Error reading file: {str(e)}")
                return
                
        # Check if content looks like a CSV
        if not any(separator in file_content for separator in [",", "\t", ";"]):
            await processing_status.edit_text(
                "The uploaded file doesn't appear to be in CSV format. Expected comma, tab, or semicolon delimiters."
            )
            return
            
        # Try different delimiters
        for delimiter in [",", "\t", ";"]:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter=delimiter)
                    if not reader.fieldnames:
                        continue
                        
                    # Check for required headers
                    required_headers = ["text", "link"]
                    if all(header in reader.fieldnames for header in required_headers):
                        # Found a working delimiter with required headers
                        rows_parsed = list(reader)  # Convert to list to preserve after file close
                        await processing_status.edit_text(f"✅ Found CSV format with {delimiter} delimiter")
                        break
            except Exception as e:
                logger.info(f"Parsing with delimiter {delimiter} failed: {str(e)}")
                continue
                
        # If we didn't parse any rows with standard headers, try alternative approaches
        if not rows_parsed:
            await processing_status.edit_text(
                "Could not find required 'text' and 'link' columns in the CSV. "
                "Please check the file format and try again."
            )
            return
            
        # Get current group ID if in a group
        current_group_id = None
        if update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP]:
            current_group_id = update.effective_chat.id
            
        # Process the successfully parsed rows
        for i, row in enumerate(rows_parsed, 1):
            try:
                # Check if we have text and link - the minimum required fields
                text = row.get("text", "").strip()
                link = row.get("link", "").strip()
                
                if not text or not link:
                    logger.warning(f"Skipping row {i}: Missing required text or link field")
                    continue
                    
                # Create entry with defaults for missing fields
                new_entry = {
                    "text": text,
                    "link": link,
                    "category": row.get("category", "General").strip() or "General",
                    "group_id": row.get("group_id", "").strip()
                }
                
                # If in a group and no group_id specified, assign current group
                if current_group_id and not new_entry["group_id"]:
                    new_entry["group_id"] = str(current_group_id)
                    
                uploaded_entries.append(new_entry)
            except Exception as row_error:
                logger.error(f"Error processing row {i}: {str(row_error)}")
                # Continue processing other rows despite this error
        # Clean up temp file at the end
        try:
            os.unlink(file_path)
        except Exception as e:
            logger.error(f"Error cleaning up temp file: {str(e)}")
        
        if not uploaded_entries:
            await processing_status.edit_text("No valid entries found in the CSV file.")
            return
        
        # Confirm before overwriting
        await processing_status.delete()
        message = f"✅ Found {len(uploaded_entries)} valid entries in the CSV file. Do you want to:"
        keyboard = [
            [
                InlineKeyboardButton("Replace All", callback_data="csv:replace"),
                InlineKeyboardButton("Append", callback_data="csv:append"),
            ],
            [InlineKeyboardButton("Cancel", callback_data="csv:cancel")]
        ]
        
        # Store entries for later use
        context.user_data["uploaded_entries"] = uploaded_entries
        
        await update.message.reply_text(message, reply_markup=InlineKeyboardMarkup(keyboard))
        
    except Exception as e:
        logger.error(f"Error processing CSV upload: {str(e)}")
        try:
            await processing_status.delete()
        except:
            pass  # Ignore if status message already deleted
        await update.message.reply_text(f"Error processing the uploaded file: {str(e)}")

async def handle_csv_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle callback queries for CSV operations."""
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
        # Determine group ID for group-specific entries
        group_id = None
        if update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP]:
            group_id = update.effective_chat.id
        
        if action == "replace":
            # If in a group, only replace entries for this group
            if group_id:
                # Keep entries that don't belong to this group
                other_entries = [entry for entry in read_entries() 
                                if entry.get("group_id") and entry.get("group_id") != str(group_id)]
                
                # Add the new entries for this group
                combined_entries = other_entries + uploaded_entries
                success = write_entries(combined_entries)
                message = f"✅ Successfully replaced {len(uploaded_entries)} entries for this group." if success else "❌ Failed to update entries."
            else:
                # In private chat, replace all entries
                success = write_entries(uploaded_entries)
                message = f"✅ Successfully replaced all entries with {len(uploaded_entries)} new entries." if success else "❌ Failed to update entries."
        
        elif action == "append":
            # Append to existing entries
            current_entries = read_entries()
            
            # Use a more comprehensive way to check for duplicates
            new_entries = []
            existing_count = 0
            
            for entry in uploaded_entries:
                # Check if this exact entry already exists
                is_duplicate = False
                for existing in current_entries:
                    if (existing["text"] == entry["text"] and 
                        existing["link"] == entry["link"] and
                        (not group_id or existing.get("group_id", "") == str(group_id))):
                        is_duplicate = True
                        existing_count += 1
                        break
                
                if not is_duplicate:
                    new_entries.append(entry)
            
            # Combine and save
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
        application.add_handler(CommandHandler("list", list_entries))  # Now admin-only
        application.add_handler(CommandHandler("add", add_entry_command))  # Still admin-only
        application.add_handler(CommandHandler("ask", ask_question))  # Available to all users
        application.add_handler(CommandHandler("download", download_csv))  # Admin-only
        application.add_handler(CommandHandler("upload", request_csv_upload))  # Admin-only
        application.add_handler(CommandHandler("clear", clear_all_entries_command))  # New admin-only command
        application.add_handler(CommandHandler("here", here_command))  # Available to all users
        application.add_handler(CommandHandler("logs", show_logs))  # New logs command
    
        # Enhanced callback query handlers
        application.add_handler(CallbackQueryHandler(handle_pagination, pattern=r"^page:"))
        application.add_handler(CallbackQueryHandler(handle_pagination, pattern=r"^delete:"))
        application.add_handler(CallbackQueryHandler(handle_pagination, pattern=r"^cat:"))
        application.add_handler(CallbackQueryHandler(handle_pagination, pattern=r"^clear:"))
        application.add_handler(CallbackQueryHandler(handle_pagination, pattern=r"^confirm_clear:"))
        application.add_handler(CallbackQueryHandler(handle_pagination, pattern=r"^cancel_clear$"))
        application.add_handler(CallbackQueryHandler(handle_csv_action, pattern=r"^csv:"))
    
        # Document handler for CSV upload - more permissive to handle different formats
        application.add_handler(
            MessageHandler(
                (filters.Document.ALL | filters.Document.FileExtension(".csv")) & 
                filters.REPLY, 
                handle_csv_upload
            )
        )
    
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
        await application.run_polling(allowed_updates=Update.ALL_TYPES)
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
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error in main process: {e}")
