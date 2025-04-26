FROM python:3.10-slim

WORKDIR /app

# Set environment variables to reduce Python buffering
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Install only necessary system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Create non-root user
RUN useradd -m -r botuser && \
    mkdir -p /app/logs && \
    touch entries.csv && \
    chown -R botuser:botuser /app

# Copy application code
COPY --chown=botuser:botuser ollama_telegram_bot.py .
COPY --chown=botuser:botuser .env .

# Switch to non-root user
USER botuser

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Command to run the application
CMD ["python", "ollama_telegram_bot.py"]
