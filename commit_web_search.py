#!/usr/bin/env python3
import subprocess
import os

# Change to the script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Run git commands
commands = [
    "git add app_db.py",
    'git commit -m "feat: add web search functionality for current information queries"',
    "git push origin main"
]

for cmd in commands:
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    print("-" * 50)

print("✅ Web search functionality has been committed and pushed!")