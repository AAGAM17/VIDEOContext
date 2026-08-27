# VideoContext Dockerfile
# Multi-stage build for minimal production image

# Build stage
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && \
    uv pip install --system -e ".[all]"

# Install system dependencies for media processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Production stage
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 videocontent && \
    mkdir -p /app/data /app/output /app/cache && \
    chown -R videocontent:videocontent /app

# Copy application code
COPY --chown=videocontent:videocontent src/ ./src/
COPY --chown=videocontent:videocontent pyproject.toml ./

# Install the package
RUN pip install --no-cache-dir -e .

USER videocontent

# Environment variables
ENV VIDEO_CONTEXT_WORKDIR=/app/data
ENV VIDEO_CONTEXT_CACHE_ENABLED=true
ENV PYTHONUNBUFFERED=1

# Default command
ENTRYPOINT ["videocontent"]
CMD ["--help"]