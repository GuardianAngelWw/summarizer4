import asyncio
import nest_asyncio
from ollama_telegram_bot import main

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Run the bot
asyncio.run(main())
