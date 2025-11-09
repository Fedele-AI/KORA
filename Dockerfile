# Multi-stage build for KORA - Knowledge Oriented Retrieval Assistant
# Supports both AMD64 and ARM64 architectures

FROM python:3.11-slim as builder

# Install libatomic for Prisma Node.js runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy UV project files
COPY pyproject.toml ./
COPY README.md LICENSE.md ./

# Copy source code and Prisma schema
COPY kora/ ./kora/
COPY prisma/ ./prisma/

# Install dependencies using UV (without lockfile)
RUN uv sync --no-dev

# Generate Prisma client
RUN uv run prisma generate

# ============================================
# Final stage - minimal runtime image
# ============================================
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# Install UV in final image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create non-root user for security
RUN useradd -m -u 1000 kora && \
    mkdir -p /app /data /models && \
    chown -R kora:kora /app /data /models

# Set working directory
WORKDIR /app

# Copy from builder
COPY --from=builder --chown=kora:kora /app ./

# Copy additional files
COPY --chown=kora:kora README.md LICENSE.md ./
COPY --chown=kora:kora config/ ./config/

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    HOME=/data \
    KORA_DATA_DIR=/data \
    KORA_MODEL_DIR=/models \
    OLLAMA_HOST=http://host.docker.internal:11434

# Switch to non-root user
USER kora

# Create necessary directories in /data
RUN mkdir -p /data/.kora /data/.kora/logs

# Expose ports
# 7860: Main Gradio UI
# 7861: Admin Gradio UI
# 8000: REST API
EXPOSE 7860 7861 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Copy entrypoint script
COPY --chown=kora:kora docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Default command
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["web"]
