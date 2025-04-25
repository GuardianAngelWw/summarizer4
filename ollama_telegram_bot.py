import os
import csv
import asyncio
import logging
import re
import tempfile
import json
from typing import List, Dict, Optional, Tuple, Any, Set
from functools import wraps

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
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
BOT_TOKEN = "6614402193:AAFm04CKorvjOeFrntJwkkqKygt1h2PGxQs"

# Admin user IDs (comma-separated in env var)
ADMIN_USER_IDS = os.getenv("ADMIN_USER_IDS", "6691432218, 5980915474").strip()
ADMIN_USERS = [int(uid.strip()) for uid in ADMIN_USER_IDS.split(",")] if ADMIN_USER_IDS else []

# CSV file to store entries
ENTRIES_FILE = "entries.csv"
CATEGORIES_FILE = "categories.json"
CSV_HEADERS = ["text", "link", "category", "group_id"]  # Added category and group_id fields

# LLM Configuration
MODEL_NAME = "tiiuae/falcon-rw-1b"  # Changed to a smaller model that works better
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Log the model loading
logger.info(f"Bot started with model: {MODEL_NAME} on device: {DEVICE}")

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

async def load_llm():
    """Load the LLM model."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    return {
        "model": model,
        "tokenizer": tokenizer,
    }

# Command Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    chat_id = update.effective_chat.id
    is_user_admin = await is_admin(context, chat_id, user.id)
    
    help_text = (
        f"👋 Hi {user.mention_html()}! I'm Summarizer2, an AI-powered bot for managing knowledge entries.\n\n"
        "Available commands:\n"
        "/ask <your question> - Ask a question about the stored entries\n"
        "/here <your question> - Answer a question (when replying to someone)\n"
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
        # Load LLM components
        await thinking_message.edit_text("Loading LLM model...")
        
        llm_components = await load_llm()
        model = llm_components["model"]
        tokenizer = llm_components["tokenizer"]
        
        await thinking_message.edit_text("Processing your question with the LLM...")
        
        # Create context from entries with categories
        context_entries = []
        for entry in entries:
            category = entry.get("category", "General")
            entry_text = f"Category: {category}\nEntry: {entry['text']}\nSource: {entry['link']}"
            context_entries.append(entry_text)
        
        context_text = "\n\n".join(context_entries)
        
        # Formulate an improved prompt
        prompt = f """You are Summarizer2, a helpful AI assistant that answers questions based on a knowledge base.

Question: "{question}"

Knowledge Base:
{context_text}

Please provide a detailed answer to the question based only on the information in the knowledge base. 
If the knowledge base doesn't contain relevant information to answer the question, say so politely.
Include relevant source links at the end of your response.
Organize the answer in a clear, easy-to-read format and in max 50 words and one paragraph also hyperlink according to telegram markdown the relevant source link to a part of word (most relevant) while sending the output.
"""
        
        # Generate response
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=1024,
            do_sample=True,
            temperature=0.7,  # Add some creativity but keep focused
            top_p=0.9,        # Focus on more likely tokens
        )
        
        result = pipe(prompt)[0]["generated_text"]
        
        # Extract the actual answer (remove the prompt)
        answer_parts = result.split("Please provide a detailed answer")
        if len(answer_parts) > 1:
            answer = answer_parts[1]
        else:
            answer = result[len(prompt):]
        
        # Clean up the answer
        if not answer.strip():
            answer = "I couldn't generate a proper response based on the available knowledge."
        
        # Send response to the replied user
        await thinking_message.delete()
        await replied_msg.reply_text(
            f"{replied_user.mention_html()}, here's the answer to: {question}\n\n{answer}",
            parse_mode=ParseMode.MARKDOWN
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

async def handle_pagination(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle pagination callbacks and other inline button actions."""
    query = update.callback_query
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    
    # Verify admin permissions
    is_user_admin = await is_admin(context, chat_id, user_id)
    if not is_user_admin:
        await query.answer("You don't have permission for this action.")
        return
    
    await query.answer()
    data = query.data
    
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
        
        # Update message
        await query.edit_message_text(message, reply_markup=reply_markup)
    
    # Handle category selection
    elif data.startswith("cat:"):
        category = data.split(":")[1]
        context.user_data['page'] = 0  # Reset page when changing category
        context.user_data['category'] = category
        
        # Re-trigger list with the selected category
        fake_update = Update(update.update_id, message=update.effective_message)
        await list_entries(fake_update, context)
    
    # Handle entry deletion
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
    
    # Handle clear confirmation
    elif data.startswith("confirm_clear:"):
        category_filter = data.split(":")[1]
        category = None if category_filter == 'all' else category_filter
        
        # Determine group ID for group-specific entries
        group_id = None
        if update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP]:
            group_id = update.effective_chat.id
        
        # Clear entries
        count = clear_all_entries(group_id, category)
        
        if count > 0:
            message = f"✅ Cleared {count} entries"
            if category:
                message += f" from category '{category}'"
            await query.edit_message_text(message)
        else:
            await query.edit_message_text("❌ No entries were cleared or an error occurred.")
    
    # Handle cancel clear
    elif data == "cancel_clear":
        await query.edit_message_text("Operation canceled.")

async def ask_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Answer a question using the LLM and knowledge base."""
    question = " ".join(context.args)
    
    if not question:
        await update.message.reply_text(
            "Please provide a question after the /ask command. For example:\n"
            "/ask What are the main features of Summarizer2?"
        )
        return
    
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
        await update.message.reply_text("No knowledge entries found to answer your question.")
        return
    
    try:
        # Load LLM components
        await thinking_message.edit_text("Loading LLM model...")
        
        llm_components = await load_llm()
        model = llm_components["model"]
        tokenizer = llm_components["tokenizer"]
        
        await thinking_message.edit_text("Processing your question with the LLM...")
        
        # Create context from entries with categories
        context_entries = []
        for entry in entries:
            category = entry.get("category", "General")
            entry_text = f"Category: {category}\nEntry: {entry['text']}\nSource: {entry['link']}"
            context_entries.append(entry_text)
        
        context_text = "\n\n".join(context_entries)
        
        # Formulate an improved prompt
        prompt = f"""You are Summarizer2, a helpful AI assistant that answers questions based on a knowledge base.

Question: "{question}"

Knowledge Base:
{context_text}

Please provide a detailed answer to the question based only on the information in the knowledge base. 
If the knowledge base doesn't contain relevant information to answer the question, say so politely.
Include relevant source links at the end of your response.
Organize the answer in a clear, easy-to-read format.
"""
        
        # Generate response
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=1024,
            temperature=0.7,  # Add some creativity but keep focused
            top_p=0.9,        # Focus on more likely tokens
        )
        
        result = pipe(prompt)[0]["generated_text"]
        
        # Extract the actual answer (remove the prompt)
        answer_parts = result.split("Please provide a detailed answer")
        if len(answer_parts) > 1:
            answer = answer_parts[1]
        else:
            answer = result[len(prompt):]
        
        # Clean up the answer
        if not answer.strip():
            answer = "I couldn't generate a proper response based on the available knowledge."
        
        # Send response
        await thinking_message.delete()
        await update.message.reply_text(
            f"Question: {question}\n\n{answer}",
            parse_mode=ParseMode.MARKDOWN
        )
    
    except Exception as e:
        logger.error(f"Error generating LLM response: {str(e)}")
        await thinking_message.delete()
        await update.message.reply_text(
            f"Sorry, I encountered an error while processing your question.\n"
            f"Error: {str(e)[:100]}..."
        )

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

def main() -> None:
    """Start the bot."""
    # Create the Application
    application = Application.builder().token(BOT_TOKEN).build()

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

    # Start the Bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
