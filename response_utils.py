from typing import Optional
import logging
from groq import AsyncGroq
import asyncio
from prompt_utils import build_prompt_with_context

logger = logging.getLogger(__name__)

async def generate_chunked_response(prompt: str, groq_client: AsyncGroq, model: str) -> str:
    """Generate a response for a single prompt chunk."""
    try:
        chat_completion = await groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200,
            top_p=0.95,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in generate_chunked_response: {str(e)}")
        return ""

async def generate_response(question: str, context_entries: list, model: str = "llama3-70b-8192") -> str:
    """
    Generate a response using chunking to handle token limits.
    """
    try:
        client = AsyncGroq(api_key=GROQ_API_KEY)
        
        # Build prompts with chunked context
        prompts = build_prompt_with_context(question, context_entries)
        
        # Generate responses for each chunk
        tasks = [generate_chunked_response(prompt, client, model) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
        
        # Combine responses
        combined_response = " ".join(filter(None, responses))
        
        # Ensure the combined response isn't too long
        if len(combined_response) > 1000:
            combined_response = combined_response[:997] + "..."
            
        return combined_response
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}")
        raise RuntimeError(f"Failed to generate response: {str(e)}")
