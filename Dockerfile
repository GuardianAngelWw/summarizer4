FROM python:3.10-slim

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    CURRENT_DATE="2025-04-27 09:59:52" \
    CURRENT_USER="GuardianAngelWw"

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
# Force reinstall to ensure all dependencies are up to date
RUN pip install --no-cache-dir -r requirements.txt --upgrade

# Setup user and directories
RUN useradd -m -r botuser && \
    mkdir -p /app/logs && \
    touch entries.csv && \
    chown -R botuser:botuser /app

# Copy application files
COPY --chown=botuser:botuser ollama_telegram_bot.py .env ./

# Switch to non-root user
USER botuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Run application with error logging
CMD ["python", "-u", "ollama_telegram_bot.py"]
