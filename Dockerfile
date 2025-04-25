FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ollama_telegram_bot.py .

# Ensure we have a directory for the entries file
RUN touch entries.csv
RUN mkdir -p /app/logs  # Create a directory for logs

# Expose a port for containerized services (optional; adjust as needed)
EXPOSE 8080

# Command to run the application
CMD ["python", "ollama_telegram_bot.py"]
