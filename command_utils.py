async def ask_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle the /ask command with improved chunking and error handling.
    """
    question = " ".join(context.args)
    if not question:
        await update.message.reply_text(
            "Please provide a question after the /ask command. For example:\n"
            "/ask Whose birthdays are in the month of April?"
        )
        return

    # Send initial thinking message
    thinking_message = await update.message.reply_text("💭")

    try:
        # Load knowledge base
        group_id = update.effective_chat.id if update.effective_chat.type in [ChatType.GROUP, ChatType.SUPERGROUP] else None
        entries = read_entries(group_id=group_id)
        
        if not entries:
            await thinking_message.delete()
            await update.message.reply_text("No knowledge entries found to answer your question.")
            return

        # Generate response with chunking
        answer = await generate_response(question, entries)
        
        # Add hyperlinks
        keywords = {entry["text"]: entry["link"] for entry in entries}
        final_answer = add_hyperlinks(answer, keywords)

        await thinking_message.delete()
        await update.message.reply_html(
            final_answer,
            disable_web_page_preview=True
        )
        
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}")
        error_message = str(e)
        if "413" in error_message:
            error_message = "The request was too large. I'll try to break it down into smaller chunks next time."
        await thinking_message.delete()
        await update.message.reply_text(f"Sorry, I encountered an error: {error_message}")
