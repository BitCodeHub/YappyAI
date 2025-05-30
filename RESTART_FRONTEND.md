# Frontend Update Instructions

The frontend has been updated from "Neo" to "Yappy". To see the changes, you need to restart the frontend.

## How to Restart the Frontend

### If frontend is running in Docker:
1. Stop the current containers:
   ```bash
   docker-compose down
   ```

2. Restart them:
   ```bash
   docker-compose up -d
   ```

### If frontend is running manually:
1. Find the frontend process (usually running on port 3000)
2. Stop it with Ctrl+C
3. Restart it:
   ```bash
   cd frontend/agentic-seek-front
   npm start
   ```

### If you're not sure how the frontend is running:
1. Check if Docker containers are running:
   ```bash
   docker ps
   ```

2. If you see containers, use the Docker method above
3. If not, the frontend might be running manually

## What Changed:
- ✅ App title: "Neo" → "Yappy"
- ✅ Welcome message: "Hello, I'm Neo" → "Hello, I'm Yappy"
- ✅ Avatar: "N" → "Y"
- ✅ Input placeholder: "Ask Neo anything" → "Ask Yappy anything"
- ✅ Browser tab title: "AgenticSeek" → "Yappy"
- ✅ Token storage: "neo_token" → "yappy_token"

After restarting, refresh your browser at http://localhost:3000 to see "Yappy" instead of "Neo"!