#!/usr/bin/env bash
set -e
PORT=${1:-8000}

# Activate venv
source venv/bin/activate

# Kill anything on PORT
if lsof -i :$PORT >/dev/null 2>&1; then
  PID=$(lsof -ti :$PORT)
  echo "Killing process on port $PORT (PID: $PID)"
  kill -9 $PID
fi

# Start server with reload
echo "Starting server on 0.0.0.0:$PORT with auto-reload..."
uvicorn server:app --host 0.0.0.0 --port $PORT --reload
