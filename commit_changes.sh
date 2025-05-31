#!/bin/bash

# Check git status for app_db.py
echo "Checking git status for app_db.py..."
git status app_db.py

# Check if app_db.py has modifications
if git diff --name-only | grep -q "app_db.py"; then
    echo "app_db.py has been modified"
    
    # Add the file
    echo "Adding app_db.py to staging..."
    git add app_db.py
    
    # Commit the changes
    echo "Committing changes..."
    git commit -m "Fix web search with DuckDuckGo priority and better error handling

- DuckDuckGo now primary search method for reliability
- Improved error handling shows when search fails
- Added debug logging to track search execution
- Weather and factual queries now properly trigger web search
- System prompts updated to handle search results correctly"
    
    # Push to GitHub
    echo "Pushing to GitHub..."
    git push origin main
    
    echo "Done! Changes pushed to GitHub."
else
    echo "app_db.py has no modifications to commit"
    
    # Check if it's already staged
    if git diff --staged --name-only | grep -q "app_db.py"; then
        echo "app_db.py is already staged, committing..."
        git commit -m "Fix web search with DuckDuckGo priority and better error handling

- DuckDuckGo now primary search method for reliability
- Improved error handling shows when search fails
- Added debug logging to track search execution
- Weather and factual queries now properly trigger web search
- System prompts updated to handle search results correctly"
        git push origin main
        echo "Done! Changes pushed to GitHub."
    fi
fi