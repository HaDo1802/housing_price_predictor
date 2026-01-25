# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port for API
EXPOSE 8000

# Run inference API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]



# ==========================================
# Multi-Stage Dockerfile for ML Application
# ==========================================
# Stage 1: Base - Common dependencies
# Stage 2: Production - Streamlit app
# ==========================================

# ==========================================
# Stage 1: Base Image with Dependencies
# ==========================================
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 mluser && \
    mkdir -p /app && \
    chown -R mluser:mluser /app

WORKDIR /app

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ==========================================
# Stage 2: Production Image
# ==========================================
FROM base as production

# Copy application code
COPY --chown=mluser:mluser . .

# Switch to non-root user
USER mluser

# Create necessary directories
RUN mkdir -p /app/logs /app/data/processed

# Expose Streamlit default port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Set the entrypoint
ENTRYPOINT ["streamlit", "run"]

# Default command (can be overridden)
CMD ["app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--server.enableCORS=false", \
    "--server.enableXsrfProtection=true", \
    "--browser.gatherUsageStats=false"]