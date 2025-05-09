FROM python:3.10-slim

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PORT=8080

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --upgrade \
    gunicorn==20.1.0 \
    google-cloud-firestore==2.11.1

# Setup user and directories
RUN useradd -m -r botuser && \
    mkdir -p /app/data/logs && \
    chown -R botuser:botuser /app

# Copy application files
COPY --chown=botuser:botuser . .

# Switch to non-root user for security
USER botuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Expose port
EXPOSE $PORT

# Run webhook handler by default
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 webhook_handler:app