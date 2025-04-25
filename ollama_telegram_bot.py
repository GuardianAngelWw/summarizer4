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
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("No BOT_TOKEN found in environment variables")

ADMIN_USER_IDS = os.getenv("ADMIN_USER_IDS", "").strip()
ADMIN_USERS = [int(uid.strip()) for uid in ADMIN_USER_IDS.split(",")] if ADMIN_USER_IDS else []

ENTRIES_FILE = "entries.csv"
CATEGORIES_FILE = "categories.json"
CSV_HEADERS = ["text", "link", "category", "group_id"]  # Added category and group_id fields

MODEL_NAME = "tiiuae/falcon-7b-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
        chat_member = await context.bot.get_chat_member(chat_id, user_id)
        return user_id in ADMIN_USERS or chat_member.status in [ChatMember.ADMINISTRATOR, ChatMember.OWNER, ChatMember.CREATOR]
    except Exception as e:
        logger.error(f"Error checking admin status: {str(e)}")
        return False

def admin_only(func):
    """Decorator to restrict command access to admin users only"""
    @wraps(func)
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        if not await is_admin(context, chat_id, user_id):
            await update.message.reply_text("Sorry, this command is restricted to group admins only.")
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
    """Add a new entry to the CSV file, preventing duplicates."""
    entries = read_entries()
    # Prevent duplicates (same text, link, category, group)
    for entry in entries:
        if entry["text"] == text and entry["link"] == link and entry.get("category", "General") == category and (str(group_id) == entry.get("group_id", "")):
            return False
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
    """Clear all entries, optionally filtered by group_id and/or category. Returns the number of entries deleted."""
    all_entries = read_entries()
    if group_id is None and category is None:
        count = len(all_entries)
        return count if write_entries([]) else 0
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
    entries = read_entries(group_id=group_id, category=category)
    if not query:
        return entries
    query = query.lower()
    return [entry for entry in entries if query in entry["text"].lower() or query in entry.get("category", "").lower()]

async def load_llm():
    """Load the LLM model."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    return {
        "model": model,
        "tokenizer": tokenizer,
    }

# --- Command Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    chat_id = update.effective_chat.id
    is_user_admin = await is_admin(context, chat_id, user.id)
    help_text = (
        f"👋 Hi {user.mention_html()}! I'm Summarizer2, an AI-powered bot for managing knowledge entries.\n\n"
        "Commands available:\n"
        "/ask <your question> — Ask a question about the stored entries\n"
    )
    if is_user_admin:
        admin_text = (
            "/list — List knowledge entries (admin only)\n"
            "/add \"entry text\" \"message_link\" \"category\" — Add a new entry\n"
            "/download — Download the current CSV file\n"
            "/upload — Upload a CSV file\n"
            "/clear — Clear all entries or entries in a specific category\n"
            "/here — Reply to someone and delete your command\n"
        )
        help_text += admin_text
    help_text += "\nUse categories to organize your knowledge."
    await update.message.reply_html(help_text)

async def here_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    if not await is_admin(context, chat_id, user_id):
        await update.message.reply_text("Sorry, this command is restricted to admins only.")
        return
    if update.message.reply_to_message is None:
        await update.message.reply_text("This command must be used as a reply to another message.")
        return
    replied_msg = update.message.reply_to_message
    replied_user = replied_msg.from_user
    # Extract after /here
    command_text = update.message.text
    response_text = command_text[5:].strip()
    if not response_text:
        response_text = "Here's the information you requested."
    await replied_msg.reply_text(
        f"{replied_user.mention_html()}, {response_text}",
        parse_mode=ParseMode.HTML
    )
    try:
        await update.message.delete()
    except Exception as e:
        logger.error(f"Error deleting message: {str(e)}")

@admin_only
async def add_entry_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Regex for quoted fields, allows nested double quotes by CSV style
    text = update.message.text[5:].strip()
    fields = []
    regex = r'("([^"]|"")*")'
    for match in re.finditer(regex, text):
        value = match.group(0)[1:-1].replace('""', '"')
        fields.append(value)
    if len(fields) < 2:
        await update.message.reply_text("Format: /add \"entry text\" \"message_link\" \"optional_category\"")
        return
    entry_text = fields[0]
    link = fields[1]
    category = fields[2] if len(fields) > 2 else "General"
    group_id = update.effective_chat.id if update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP] else None
    if add_entry(entry_text, link, category, group_id):
        await update.message.reply_text(f"✅ Added entry:\nCategory: {category}\nText: {entry_text}\nLink: {link}")
    else:
        await update.message.reply_text("❌ Entry already exists or could not be added (duplicate prevention is enabled).")

@admin_only
async def list_entries(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args if context.args else []
    query = ""
    category = None
    for arg in args:
        if arg.startswith("category="):
            category = arg.split("=", 1)[1]
        else:
            query = arg
    page = int(context.user_data.get('page', 0))
    group_id = update.effective_chat.id if update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP] else None
    entries = search_entries(query, group_id, category) if query else read_entries(group_id, category)
    total_pages = (len(entries) + ENTRIES_PER_PAGE - 1) // ENTRIES_PER_PAGE
    start_idx = page * ENTRIES_PER_PAGE
    end_idx = min(start_idx + ENTRIES_PER_PAGE, len(entries))
    if not entries:
        message = "No entries found."
        if category: message += f" in category '{category}'"
        if query: message += f" matching '{query}'"
        await update.message.reply_text(message)
        return
    message = f"📚 <b>Entries {start_idx+1}-{end_idx} of {len(entries)}</b>"
    if category: message += f" in <i>{category}</i>"
    if query: message += f" matching '<code>{query}</code>'"
    message += ":\n\n"
    for i, entry in enumerate(entries[start_idx:end_idx], start=start_idx + 1):
        message += f"<b>{i}.</b> <code>[{entry.get('category', 'General')}]</code> {entry['text']}\n"
        message += f"   <a href='{entry['link']}'>🔗 Link</a>\n\n"
    categories = get_categories()
    keyboard = []
    cat_row = [InlineKeyboardButton(f"📂 {cat}", callback_data=f"cat:{cat}") for cat in categories[:3]]
    if cat_row: keyboard.append(cat_row)
    nav_row = []
    if page > 0:
        nav_row.append(InlineKeyboardButton("◀️ Prev", callback_data=f"page:{page-1}:{category or ''}"))
    if page < total_pages - 1:
        nav_row.append(InlineKeyboardButton("Next ▶️", callback_data=f"page:{page+1}:{category or ''}"))
    if nav_row: keyboard.append(nav_row)
    for i in range(start_idx, end_idx):
        keyboard.append([InlineKeyboardButton(f"🗑️ Delete #{i+1}", callback_data=f"delete:{i}")])
    if entries:
        clear_text = f"Clear '{category}'" if category else "Clear All"
        keyboard.append([InlineKeyboardButton(f"🗑️ {clear_text}", callback_data=f"clear:{category or 'all'}")])
    reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
    context.user_data['page'] = page
    context.user_data['category'] = category
    await update.message.reply_html(message, reply_markup=reply_markup, disable_web_page_preview=True)

async def handle_pagination(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    if not await is_admin(context, chat_id, user_id):
        await query.answer("You don't have permission for this action.")
        return
    await query.answer()
    data = query.data
    if data.startswith("page:"):
        parts = data.split(":")
        page = int(parts[1])
        category = parts[2] if len(parts) > 2 and parts[2] else None
        context.user_data['page'] = page
        group_id = chat_id if update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP] else None
        entries = read_entries(group_id, category)
        total_pages = (len(entries) + ENTRIES_PER_PAGE - 1) // ENTRIES_PER_PAGE
        start_idx = page * ENTRIES_PER_PAGE
        end_idx = min(start_idx + ENTRIES_PER_PAGE, len(entries))
        message = f"📚 <b>Entries {start_idx+1}-{end_idx} of {len(entries)}</b>"
        if category: message += f" in <i>{category}</i>"
        message += ":\n\n"
        for i, entry in enumerate(entries[start_idx:end_idx], start=start_idx + 1):
            message += f"<b>{i}.</b> <code>[{entry.get('category','General')}]</code> {entry['text']}\n"
            message += f"   <a href='{entry['link']}'>🔗 Link</a>\n\n"
        categories = get_categories()
        keyboard = []
        cat_row = [InlineKeyboardButton(f"📂 {cat}", callback_data=f"cat:{cat}") for cat in categories[:3]]
        if cat_row: keyboard.append(cat_row)
        nav_row = []
        if page > 0:
            nav_row.append(InlineKeyboardButton("◀️ Prev", callback_data=f"page:{page-1}:{category or ''}"))
        if page < total_pages - 1:
            nav_row.append(InlineKeyboardButton("Next ▶️", callback_data=f"page:{page+1}:{category or ''}"))
        if nav_row: keyboard.append(nav_row)
        for i in range(start_idx, end_idx):
            keyboard.append([InlineKeyboardButton(f"🗑️ Delete #{i+1}", callback_data=f"delete:{i}")])
        if entries:
            clear_text = f"Clear '{category}'" if category else "Clear All"
            keyboard.append([InlineKeyboardButton(f"🗑️ {clear_text}", callback_data=f"clear:{category or 'all'}")])
        reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
        await query.edit_message_text(message, reply_markup=reply_markup, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    elif data.startswith("cat:"):
        category = data.split(":", 1)[1]
        context.user_data['page'] = 0
        context.user_data['category'] = category
        fake_update = Update(update.update_id, message=update.effective_message)
        await list_entries(fake_update, context)
    elif data.startswith("delete:"):
        index = int(data.split(":")[1])
        if delete_entry(index):
            await query.edit_message_text(f"✅ Entry #{index+1} deleted successfully.")
        else:
            await query.edit_message_text(f"❌ Failed to delete entry #{index+1}.")
    elif data.startswith("clear:"):
        category_filter = data.split(":")[1]
        category = None if category_filter == 'all' else category_filter
        group_id = chat_id if update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP] else None
        confirm_text = "Are you sure you want to clear "
        if category: confirm_text += f"all entries in category '<b>{category}</b>'?"
        else: confirm_text += "<b>ALL</b> entries?"
        keyboard = [
            [
                InlineKeyboardButton("Yes, Clear", callback_data=f"confirm_clear:{category or 'all'}"),
                InlineKeyboardButton("Cancel", callback_data="cancel_clear")
            ]
        ]
        await query.edit_message_text(confirm_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)
    elif data.startswith("confirm_clear:"):
        category_filter = data.split(":")[1]
        category = None if category_filter == 'all' else category_filter
        group_id = chat_id if update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP] else None
        count = clear_all_entries(group_id, category)
        if count > 0:
            message = f"✅ Cleared {count} entries"
            if category: message += f" from category '<b>{category}</b>'"
            await query.edit_message_text(message, parse_mode=ParseMode.HTML)
        else:
            await query.edit_message_text("❌ No entries were cleared or an error occurred.")
    elif data == "cancel_clear":
        await query.edit_message_text("Operation canceled.")

async def ask_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    question = " ".join(context.args)
    if not question:
        await update.message.reply_text(
            "Please provide a question after the /ask command. For example:\n"
            "/ask What are the main features of Summarizer2?"
        )
        return
    await update.message.reply_text("🤔 Thinking about your question... This might take a moment.")
    entries = read_entries()
    if not entries:
        await update.message.reply_text("No knowledge entries found to answer from.")
        return
    try:
        progress_message = await update.message.reply_text("Loading LLM model...")
        llm_components = await load_llm()
        model = llm_components["model"]
        tokenizer = llm_components["tokenizer"]
        await progress_message.edit_text("Processing your question with the LLM...")
        context_text = "\n\n".join([f"Entry: {entry['text']}\nSource: {entry['link']}" for entry in entries])
        prompt = f"""Based on the following knowledge entries, answer this question: "{question}"\nKnowledge Base: {context_text}\nGive a comprehensive answer based on the provided knowledge. If the knowledge is insufficient, say so."""
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=1024,
        )
        result = pipe(prompt)[0]["generated_text"]
        answer = result.split("Give a comprehensive answer")[-1].strip() if "Give a comprehensive answer" in result else result
        await progress_message.delete()
        await update.message.reply_text(
            f"Question: {question}\n\n{answer}",
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception as e:
        logger.error(f"Error generating LLM response: {str(e)}")
        await update.message.reply_text(
            f"Sorry, I encountered an error while processing your question.\n"
            f"Error: {str(e)[:100]}..."
        )

@admin_only
async def clear_all_entries_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    category = None
    if context.args:
        category = " ".join(context.args)
    group_id = update.effective_chat.id if update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP] else None
    confirm_text = "Are you sure you want to clear "
    if category: confirm_text += f"all entries in category '<b>{category}</b>'?"
    else: confirm_text += "<b>ALL</b> entries?"
    keyboard = [
        [
            InlineKeyboardButton("Yes, Clear", callback_data=f"confirm_clear:{category or 'all'}"),
            InlineKeyboardButton("Cancel", callback_data="cancel_clear")
        ]
    ]
    await update.message.reply_html(confirm_text, reply_markup=InlineKeyboardMarkup(keyboard))

@admin_only
async def download_csv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not os.path.exists(ENTRIES_FILE):
        await update.message.reply_text("No entries file exists yet.")
        return
    group_id = update.effective_chat.id if update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP] else None
    if group_id:
        entries = read_entries(group_id)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            try:
                with open(temp_file.name, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                    writer.writeheader()
                    writer.writerows(entries)
                await update.message.reply_document(
                    document=open(temp_file.name, "rb"),
                    filename=f"entries_{update.effective_chat.title or 'group'}.csv",
                    caption=f"Here's your knowledge base CSV file for this group. ({len(entries)} entries)"
                )
            finally:
                os.unlink(temp_file.name)
    else:
        await update.message.reply_document(
            document=open(ENTRIES_FILE, "rb"),
            filename="entries.csv",
            caption=f"Here's your complete knowledge base CSV file."
        )

@admin_only
async def request_csv_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    categories = get_categories()
    category_list = ", ".join(categories)
    await update.message.reply_text(
        "Please upload your CSV file as a reply to this message.\n\n"
        f"The file should have these columns: 'text', 'link', 'category', 'group_id'\n\n"
        f"Available categories: {category_list}\n\n"
        "If you're adding entries for this group, leave group_id empty."
    )
    context.user_data["awaiting_csv"] = True

@admin_only
async def handle_csv_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.user_data.get("awaiting_csv"):
        return
    context.user_data["awaiting_csv"] = False
    if not update.message.document:
        await update.message.reply_text("Please upload a CSV file.")
        return
    document = update.message.document
    file_name = document.file_name.lower()
    # Accept only .csv extension and proper MIME type
    if not (file_name.endswith(".csv") or "csv" in (document.mime_type or "")):
        await update.message.reply_text("Please upload a file with .csv extension.")
        return
    try:
        file = await context.bot.get_file(document.file_id)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            await file.download_to_drive(temp_file.name)
            uploaded_entries = []
            try:
                with open(temp_file.name, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    actual_headers = reader.fieldnames or []
                    required_headers = ["text", "link"]
                    if not all(header in actual_headers for header in required_headers):
                        await update.message.reply_text(
                            f"CSV must contain at least these headers: {', '.join(required_headers)}"
                        )
                        return
                    current_group_id = update.effective_chat.id if update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP] else None
                    for row in reader:
                        if row.get("text") and row.get("link"):
                            new_entry = {
                                "text": row["text"],
                                "link": row["link"],
                                "category": row.get("category", "General"),
                                "group_id": row.get("group_id", "")
                            }
                            if current_group_id and not new_entry["group_id"]:
                                new_entry["group_id"] = str(current_group_id)
                            uploaded_entries.append(new_entry)
            finally:
                os.unlink(temp_file.name)
        if not uploaded_entries:
            await update.message.reply_text("No valid entries found in the CSV file.")
            return
        message = f"Found {len(uploaded_entries)} valid entries in the CSV file. Do you want to:"
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
        group_id = update.effective_chat.id if update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP] else None
        if action == "replace":
            if group_id:
                other_entries = [entry for entry in read_entries() if entry.get("group_id") and entry.get("group_id") != str(group_id)]
                combined_entries = other_entries + uploaded_entries
                success = write_entries(combined_entries)
                message = f"✅ Successfully replaced {len(uploaded_entries)} entries for this group." if success else "❌ Failed to update entries."
            else:
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
                        existing["link"] == entry["link"] and
                        (not group_id or existing.get("group_id", "") == str(group_id))):
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

def main() -> None:
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", start))
    application.add_handler(CommandHandler("list", list_entries))
    application.add_handler(CommandHandler("add", add_entry_command))
    application.add_handler(CommandHandler("ask", ask_question))
    application.add_handler(CommandHandler("download", download_csv))
    application.add_handler(CommandHandler("upload", request_csv_upload))
    application.add_handler(CommandHandler("clear", clear_all_entries_command))
    application.add_handler(CommandHandler("here", here_command))
    application.add_handler(CallbackQueryHandler(handle_pagination, pattern=r"^page:"))
    application.add_handler(CallbackQueryHandler(handle_pagination, pattern=r"^delete:"))
    application.add_handler(CallbackQueryHandler(handle_pagination, pattern=r"^cat:"))
    application.add_handler(CallbackQueryHandler(handle_pagination, pattern=r"^clear:"))
    application.add_handler(CallbackQueryHandler(handle_pagination, pattern=r"^confirm_clear:"))
    application.add_handler(CallbackQueryHandler(handle_pagination, pattern=r"^cancel_clear$"))
    application.add_handler(CallbackQueryHandler(handle_csv_action, pattern=r"^csv:"))
    application.add_handler(
        MessageHandler(
            (filters.Document.MimeType("text/csv") | 
             filters.Document.FileExtension(".csv")) & 
            filters.REPLY, 
            handle_csv_upload
        )
    )
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
