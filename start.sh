#!/bin/bash
# start.sh
# Starts the repolens FastAPI backend and React frontend together.
# Run from the project root after completing setup in README.md.
#
# Usage: bash start.sh

set -e

echo "Starting repolens..."
echo ""

# Verify environment
if [ -z "$OPENAI_API_KEY" ] && [ ! -f ".env" ]; then
  echo "Error: No .env file found and OPENAI_API_KEY is not set."
  echo "Create a .env file with OPENAI_API_KEY=sk-your-key-here"
  exit 1
fi

# Install frontend dependencies if missing
if [ ! -d "frontend/node_modules" ]; then
  echo "Frontend dependencies not installed. Running npm install..."
  cd frontend && npm install && cd ..
fi

# Start FastAPI backend in background
echo "Starting FastAPI backend at http://localhost:8000"
uvicorn repolens.api:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

sleep 2

# Start React frontend in background
echo "Starting React frontend at http://localhost:3000"
cd frontend
npm run dev -- --port 3000 &
FRONTEND_PID=$!
cd ..

echo ""
echo "repolens is running."
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:3000"
echo "  Docs:     http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers."

# Clean shutdown on Ctrl+C or SIGTERM.
# Without this trap, Ctrl+C kills the shell but leaves uvicorn
# and Vite running as orphaned background processes.
trap "echo ''; echo 'Shutting down...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait $BACKEND_PID $FRONTEND_PID
