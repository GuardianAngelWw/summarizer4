import os
import logging
import json
from flask import Flask, request, jsonify
import telegram
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters
)
from ollama_telegram_bot import (
    start, here_command, ask_question, list_entries, add_entry_command,
    download_csv, request_csv_upload, clear_all_entries_command, show_logs,
    show_entry_command, insert_entry_command, slowmode_command, handle_pagination,
    handle_single_entry_delete, handle_csv_action, set_prompt_command,
    set_model_command, set_apikey_command, BotStatusMonitor, load_dotenv
)
import nest_asyncio
import asyncio
from dotenv import load_dotenv

# Apply nest_asyncio to handle async operations in webhook mode
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize Flask app 
app = Flask(__name__)

# Get environment variables with defaults
PORT = int(os.environ.get("PORT", 8080))
BOT_TOKEN = os.environ.get("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN environment variable is not set!")

# Store application instance globally
app_instance = None

# Create data directories
os.makedirs('data', exist_ok=True)
os.makedirs('data/logs', exist_ok=True)

@app.route(f"/{BOT_TOKEN}", methods=["POST"])
async def webhook_handler():
    """Handle incoming webhook updates from Telegram."""
    if request.method == "POST":
        try:
            update_data = request.get_json(force=True)
            logger.debug(f"Received update: {json.dumps(update_data)}")
            
            # Process update asynchronously
            if app_instance:
                await app_instance.update_queue.put(Update.de_json(data=update_data, bot=app_instance.bot))
                return jsonify({"status": "success"})
            else:
                logger.error("Application instance not initialized")
                return jsonify({"status": "error", "message": "Bot not initialized"}), 500
        except Exception as e:
            logger.error(f"Error processing update: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify({"status": "error", "message": "Method not allowed"}), 405

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for Cloud Run."""
    try:
        if app_instance and app_instance.bot:
            return jsonify({
                "status": "healthy",
                "bot_name": app_instance.bot.username,
                "timestamp": str(telegram.utils.helpers.utc_now())
            })
        return jsonify({
            "status": "starting",
            "timestamp": str(telegram.utils.helpers.utc_now())
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": str(telegram.utils.helpers.utc_now())
        }), 500

@app.route("/", methods=["GET"])
def index():
    """Root endpoint with basic info."""
    return jsonify({
        "service": "Telegram Bot Webhook Handler",
        "status": "running",
        "documentation": "/health for health check endpoint"
    })

async def setup_application():
    """Set up the Application instance with all handlers."""
    global app_instance
    
    # Initialize your application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Add handlers
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
    application.add_handler(CommandHandler("logs", show_logs))
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
    
    # Document handler for CSV upload
    application.add_handler(
        MessageHandler(
            (filters.Document.ALL | filters.Document.FileExtension(".csv")) & 
            filters.REPLY, 
            handle_csv_upload
        )
    )
    
    app_instance = application
    
    # Start the application in webhook mode
    webhook_url = os.environ.get("WEBHOOK_URL")
    if webhook_url:
        webhook_path = f"/{BOT_TOKEN}"
        full_webhook_url = f"{webhook_url.rstrip('/')}{webhook_path}"
        
        # Set webhook
        await application.bot.set_webhook(full_webhook_url)
        logger.info(f"Webhook set to {full_webhook_url}")
        
        # Start handling updates
        await application.start()
        logger.info("Bot started in webhook mode")
        
        # Notify admins about startup
        admin_ids = [int(id.strip()) for id in os.environ.get("ADMIN_USER_IDS", "").split(",") if id.strip()]
        status_monitor = BotStatusMonitor(BOT_TOKEN, admin_ids)
        await status_monitor.send_startup_notification()
    else:
        logger.error("WEBHOOK_URL environment variable is not set!")
        raise ValueError("WEBHOOK_URL environment variable is required for webhook mode")

def run_webhook_server():
    """Run the webhook server."""
    # Setup the application
    asyncio.run(setup_application())
    
    # Run the Flask app
    try:
        from waitress import serve
        logger.info(f"Starting webhook server with waitress on port {PORT}")
        serve(app, host="0.0.0.0", port=PORT)
    except ImportError:
        logger.info(f"Waitress not available, using Flask's built-in server on port {PORT}")
        app.run(host="0.0.0.0", port=PORT, debug=False)

if __name__ == "__main__":
    run_webhook_server()