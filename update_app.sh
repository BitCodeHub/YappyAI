#!/bin/bash

# Backup current app_db.py
echo "Backing up current app_db.py..."
cp app_db.py app_db_backup_$(date +%Y%m%d_%H%M%S).py

# Copy fixed version to app_db.py
echo "Updating app_db.py with fixed version..."
cp app_db_fixed.py app_db.py

echo "Done! app_db.py has been updated."
echo ""
echo "Now you can commit and push:"
echo "git add app_db.py"
echo "git commit -m 'Fix web search to match local app architecture'"
echo "git push origin main"