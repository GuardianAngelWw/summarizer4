FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    gcc \
    g++ \
    git \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ollama_telegram_bot.py .

# Ensure we have a directory for the entries file
RUN touch entries.csv && \
    mkdir -p /app/logs

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose a port for containerized services
EXPOSE 8080

# Command to run the application
CMD ["python", "ollama_telegram_bot.py"]
