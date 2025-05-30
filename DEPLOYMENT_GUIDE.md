# Deployment Guide for AgenticSeek with PDF Support

## Option 1: Local Deployment Package (Easiest)

### Create a standalone package:
1. **Bundle everything** into a zip file with start scripts
2. **Include instructions** for users to:
   - Install Python 3.8+
   - Run `pip install -r requirements.txt`
   - Double-click start script

### Start Scripts for Users:

**Windows (start_app.bat):**
```batch
@echo off
echo Starting AgenticSeek...
start cmd /k "python api_clean.py"
timeout /t 3
start http://localhost:3000
```

**Mac/Linux (start_app.sh):**
```bash
#!/bin/bash
echo "Starting AgenticSeek..."
python3 api_clean.py &
sleep 3
open http://localhost:3000
```

## Option 2: Docker Deployment (Recommended)

### Create Dockerfile:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose ports
EXPOSE 8000 3000

# Start command
CMD ["python", "api_clean.py"]
```

### Docker Compose for complete stack:
```yaml
version: '3'
services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - API_KEY=your_gemini_api_key
    volumes:
      - ./uploads:/app/uploads
  
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
```

## Option 3: Cloud Deployment

### Deploy to cloud services:

**1. Heroku (Free tier available):**
- Add `Procfile`: `web: uvicorn api_clean:app --host 0.0.0.0 --port $PORT`
- Deploy with Git

**2. Google Cloud Run:**
- Containerize the app
- Deploy with `gcloud run deploy`

**3. AWS EC2/Lambda:**
- Use EC2 for full control
- Lambda for serverless (with modifications)

**4. Replit/Railway:**
- Easy one-click deploy platforms
- Good for demos

## Option 4: Desktop App (Professional)

### Use PyInstaller or similar:
```bash
pip install pyinstaller
pyinstaller --onefile --windowed start_app.py
```

### Create installer:
- **Windows**: Use Inno Setup or NSIS
- **Mac**: Create .dmg with create-dmg
- **Linux**: Create .deb or .AppImage

## Security Considerations

### Before deploying:
1. **API Keys**: Move to environment variables
2. **Authentication**: Current auth system is basic - enhance for production
3. **HTTPS**: Use SSL certificates
4. **Rate Limiting**: Add to prevent abuse
5. **File Upload Limits**: Set max file sizes
6. **Sanitization**: Validate all inputs

### Production-ready changes needed:
```python
# In api_clean.py
import os
from dotenv import load_dotenv

load_dotenv()

# Replace hardcoded API key
API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not set")

# Add file size limits
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Add rate limiting
from slowapi import Limiter
limiter = Limiter(key_func=lambda: "global")
app.state.limiter = limiter

@app.post("/query")
@limiter.limit("10/minute")
async def query(request: QueryRequest):
    # ... existing code
```

## Quick Deployment Script

### create_deployment.py:
```python
#!/usr/bin/env python3
import os
import shutil
import zipfile

def create_deployment():
    # Create deployment directory
    deploy_dir = "agenticseek_deploy"
    if os.path.exists(deploy_dir):
        shutil.rmtree(deploy_dir)
    os.makedirs(deploy_dir)
    
    # Copy necessary files
    files_to_copy = [
        'api_clean.py',
        'requirements.txt',
        'start_neo_api.py',
        'PDF_UPLOAD_GUIDE.md',
        '.env.example'
    ]
    
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy(file, deploy_dir)
    
    # Copy directories
    dirs_to_copy = ['frontend', 'sources', 'prompts']
    for dir in dirs_to_copy:
        if os.path.exists(dir):
            shutil.copytree(dir, os.path.join(deploy_dir, dir))
    
    # Create .env template
    with open(os.path.join(deploy_dir, '.env'), 'w') as f:
        f.write("GEMINI_API_KEY=your_api_key_here\n")
    
    # Create README
    with open(os.path.join(deploy_dir, 'README.txt'), 'w') as f:
        f.write("""AgenticSeek Installation Instructions

1. Install Python 3.8 or higher
2. Open terminal/command prompt in this directory
3. Run: pip install -r requirements.txt
4. Edit .env file and add your Gemini API key
5. Run: python start_neo_api.py
6. Open http://localhost:3000 in your browser

For support, see PDF_UPLOAD_GUIDE.md
""")
    
    # Create zip file
    with zipfile.ZipFile('agenticseek_deploy.zip', 'w') as zipf:
        for root, dirs, files in os.walk(deploy_dir):
            for file in files:
                zipf.write(os.path.join(root, file))
    
    print("✅ Deployment package created: agenticseek_deploy.zip")

if __name__ == "__main__":
    create_deployment()
```

## Recommended Approach

For deploying to others, I recommend:

1. **For Technical Users**: Docker Compose package
2. **For Non-Technical Users**: Desktop app with installer
3. **For Teams**: Cloud deployment (Heroku/Cloud Run)
4. **For Demo/Testing**: Replit or Railway

The key is to make it as simple as possible for your target users!