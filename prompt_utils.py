from typing import List, Dict
from token_utils import count_tokens, chunk_text

def build_prompt_with_context(question: str, context_entries: List[Dict[str, str]], max_tokens: int = 5500) -> List[str]:
    """
    Build prompts with context, ensuring they don't exceed token limits.
    Returns a list of prompts if chunking is needed.
    """
    base_prompt = """You are an AI assistant with relatively high egoistic nature. Based on the provided knowledge base, \
summarize the context and provide a solution to the question without repeating it in [...]
    - While answering, ensure the answer is concise, understandable, and easy to follow and in very short brief.
    - **MUST** use Telegram HTML markdown formatting for every answer with <blockquote>from starting of the answer to the end of the answer</blockquote>
    - **MUST** use Telegram HTML markdown formatting for every answer with <a href="source link">Relevant word of the output</a>.
    - **MUST** If the question contains any NSFW-themed content, reply with "/report WB POLICE 🚓[...]
    - **MUST** read the whole question so every word of the question makes sense in the output.
    - **NEVER** mention about the knowledge base in the output or anything if you can / can't find.
    - **NEVER** reply out-of-context or out of entries questions.

Question: {question}

Knowledge Base:
"""
    # Calculate available tokens for context
    base_tokens = count_tokens(base_prompt.format(question=question))
    available_tokens = max_tokens - base_tokens - 100  # Buffer for safety

    # Create chunks of context entries
    context_chunks = []
    current_chunk = []
    current_tokens = 0

    for entry in context_entries:
        entry_text = f"Category: {entry.get('category', 'General')}\nEntry: {entry['text']}\nSource: {entry['link']}\n\n"
        entry_tokens = count_tokens(entry_text)

        if current_tokens + entry_tokens > available_tokens:
            if current_chunk:
                context_chunks.append(current_chunk)
            current_chunk = [entry_text]
            current_tokens = entry_tokens
        else:
            current_chunk.append(entry_text)
            current_tokens += entry_tokens

    if current_chunk:
        context_chunks.append(current_chunk)

    # Build final prompts
    prompts = []
    for chunk in context_chunks:
        context_text = "".join(chunk)
        prompts.append(base_prompt.format(question=question) + context_text)

    return prompts
