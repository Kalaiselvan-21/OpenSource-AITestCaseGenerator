#!/bin/bash

# Kill any existing Flask processes
pkill -f "python3 app.py" || true

# Start the API server
echo "Starting AI Test Case Generator API..."

# Start the Flask app
cd "$(dirname "$0")"
python3 app.py > app.log 2>&1 &

echo "Started Flask app on port 5002"
echo "Log file: $(pwd)/app.log"

# Wait a moment for the app to start
sleep 2

# Check if the app is running
if curl -s http://localhost:5002/health > /dev/null; then
    echo "Flask app is running successfully"
    
    # Add test tokens to verify token tracking is working
    python3 add_test_tokens.py --prompt 100 --completion 50 --stats
else
    echo "Failed to start Flask app"
    cat app.log
fi
