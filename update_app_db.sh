#!/bin/bash
# Simple script to update app_db.py with v4 features

echo "🔧 Updating app_db.py with v4 features..."

# Backup original
cp app_db.py app_db_original_backup.py
echo "✅ Created backup: app_db_original_backup.py"

# Copy v4 version to app_db.py
cp app_v4_fixed.py app_db.py
echo "✅ Replaced app_db.py with v4 version"

# Show the changes
echo ""
echo "📝 Changes applied. To deploy:"
echo "1. git add app_db.py"
echo "2. git commit -m 'fix: update app_db.py with context awareness and real-time search'"
echo "3. git push origin main"
echo ""
echo "4. Go to Render dashboard and click 'Manual Deploy' > 'Clear build cache & deploy'"