#!/bin/bash
echo "Killing existing API processes..."
ps aux | grep "python.*api_clean.py" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null
lsof -ti:8000 | xargs kill -9 2>/dev/null

echo "Starting enhanced API with advanced visualizations..."
cd /Users/jimmylam/Downloads/agenticSeek-main
nohup python3 api_clean.py > api_enhanced.log 2>&1 &

echo "API started! Check api_enhanced.log for status."