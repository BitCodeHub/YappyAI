#!/bin/bash

echo "Adding static files to git..."

# Force add static directory
git add -f static/
git add -f static/index.html
git add -f static/yappy.html

# Check status
echo "Git status of static files:"
git status static/

echo "Files in static directory:"
ls -la static/

echo "Done! Now commit and push:"
echo "git commit -m 'add: static files including yappy.html'"
echo "git push origin main"