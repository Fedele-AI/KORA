#!/bin/bash
set -e

# Ensure data directories exist
mkdir -p /data/.kora /data/.kora/logs

# Function to check if Ollama is accessible
check_ollama() {
    echo "🔍 Checking Ollama connectivity at $OLLAMA_HOST..."
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -sf "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
            echo "✅ Ollama is accessible"
            return 0
        fi
        echo "⏳ Waiting for Ollama (attempt $attempt/$max_attempts)..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "⚠️  Warning: Could not connect to Ollama at $OLLAMA_HOST"
    echo "   Make sure Ollama is running and accessible"
    echo "   Continuing anyway - you can configure this later"
    return 1
}

# Check Ollama on startup
check_ollama || true

# Initialize database if needed
echo "📊 Initializing database..."
if [ ! -z "$DATABASE_URL" ]; then
    echo "  Database URL: $DATABASE_URL"
    # Push schema to database (creates tables if they don't exist)
    uv run prisma db push --skip-generate 2>/dev/null || echo "  Database already initialized"
else
    echo "  Using default SQLite database"
fi

# Print startup banner
echo ""
echo "=========================================="
echo "  🚀 KORA Container Starting"
echo "=========================================="
echo ""
echo "Environment:"
echo "  • Data directory: $KORA_DATA_DIR"
echo "  • Model directory: $KORA_MODEL_DIR"
echo "  • Ollama host: $OLLAMA_HOST"
echo "  • Database: ${DATABASE_URL:-file:/data/kora.db}"
echo "  • DB Logging: ${ENABLE_DB_LOGGING:-true}"
echo ""

# Execute the requested service
case "$1" in
    web)
        echo "🌐 Starting KORA Web Interface..."
        echo ""
        exec uv run kora-launch --server-name 0.0.0.0 --server-port 7860 --admin-port 7861
        ;;
    api)
        echo "🔌 Starting KORA REST API..."
        echo ""
        exec uv run python -m uvicorn kora.api:app --host 0.0.0.0 --port 8000
        ;;
    admin)
        echo "🔐 Starting KORA Admin Panel..."
        echo ""
        exec uv run kora-admin --server-name 0.0.0.0 --server-port 7861
        ;;
    auth)
        echo "🔑 Running KORA Auth CLI..."
        shift
        exec uv run kora-auth "$@"
        ;;
    bash|sh)
        echo "🐚 Starting interactive shell..."
        exec /bin/bash
        ;;
    *)
        echo "Usage: docker run [OPTIONS] kora [web|api|admin|auth|bash]"
        echo ""
        echo "Services:"
        echo "  web     - Start web interface (default)"
        echo "  api     - Start REST API server"
        echo "  admin   - Start admin panel"
        echo "  auth    - Run auth CLI commands"
        echo "  bash    - Start interactive shell"
        echo ""
        echo "Examples:"
        echo "  docker run -p 7860:7860 kora web"
        echo "  docker run -p 8000:8000 kora api"
        echo "  docker run kora auth generate --username admin"
        exit 1
        ;;
esac
