#!/bin/sh
set -e

echo "🔍 Waiting for PostgreSQL to be ready..."

# Wait for PostgreSQL using wget (available in Alpine)
MAX_RETRIES=30
RETRY_COUNT=0

# Extract host and port from DATABASE_URL or use defaults
DB_HOST="${POSTGRES_HOST:-postgres}"
DB_PORT="${POSTGRES_PORT:-5432}"

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    # Try to connect using timeout and shell
    if (echo > /dev/tcp/$DB_HOST/$DB_PORT) 2>/dev/null; then
        echo "✅ PostgreSQL is ready!"
        break
    fi
    
    # Alternative check: try with node
    if node -e "require('net').createConnection($DB_PORT, '$DB_HOST', () => { process.exit(0); }).on('error', () => { process.exit(1); })" 2>/dev/null; then
        echo "✅ PostgreSQL is ready!"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "⏳ Waiting for PostgreSQL at $DB_HOST:$DB_PORT... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "⚠️ PostgreSQL connection timeout after $MAX_RETRIES attempts"
    echo "   Proceeding anyway - Prisma will retry on its own..."
fi

echo "🔄 Running database migrations..."
# Continue even if migrations fail (schema may already exist)
npx prisma migrate deploy || {
    echo "⚠️ Migration warning: continuing anyway (schema may already exist)"
}

echo "🚀 Starting NeuralTrade API..."
exec node dist/main.js
