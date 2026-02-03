#!/usr/bin/env bash
# Kill any process listening on the BCI web UI port(s). Run this if you see
# "address already in use" when starting with --web.
# Usage: ./scripts/kill_web_server.sh [port]
# Default port: 8765

PORT="${1:-8765}"
for p in $PORT 8766 8767; do
  PID=$(lsof -ti ":$p" 2>/dev/null)
  if [ -n "$PID" ]; then
    echo "Killing process(es) on port $p: $PID"
    kill $PID 2>/dev/null || kill -9 $PID 2>/dev/null
  fi
done
echo "Done. You can run: PYTHONPATH=. python main.py --web --online"
