#!/bin/bash
source venv/bin/activate
python run.py api --port 5002 > api_server.log 2>&1 &
echo $! > api_server.pid
echo "API server started on port 5002 (PID: $(cat api_server.pid))"
