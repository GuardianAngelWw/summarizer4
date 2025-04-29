import tiktoken

def count_tokens(text: str, model: str = "llama3-70b-8192") -> int:
    """
    Count the number of tokens in a text string.
    Args:
        text (str): The text to count tokens for
        model (str): The model name to use for counting
    Returns:
        int: Number of tokens
    """
    try:
        # For Llama models, use cl100k_base encoding
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback to approximate token count if tiktoken fails
        return len(text.split()) * 1.3  # Rough estimate

def chunk_text(text: str, max_tokens: int = 5500, model: str = "llama3-70b-8192") -> list[str]:
    """
    Split text into chunks that won't exceed token limit.
    Args:
        text (str): Text to split into chunks
        max_tokens (int): Maximum tokens per chunk (default 5500 to leave room for prompt)
        model (str): Model name for token counting
    Returns:
        list[str]: List of text chunks
    """
    if count_tokens(text, model) <= max_tokens:
        return [text]

    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    # Split by sentences for more natural breaks
    sentences = text.split('. ')
    
    for sentence in sentences:
        sentence = sentence.strip() + '. '
        sentence_tokens = count_tokens(sentence, model)
        
        if current_tokens + sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_tokens = sentence_tokens
        else:
            current_chunk += sentence
            current_tokens += sentence_tokens
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks
