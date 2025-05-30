import os
import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

# Initialize FastAPI with metadata
app = FastAPI(
    title="Yappy AI Assistant",
    description="Your friendly AI companion with multiple LLM support",
    version="2.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# In-memory storage (replace with database in production)
users_db = {}
conversations_db = {}
api_keys_db = {}

# Pydantic models
class UserSignup(BaseModel):
    username: str
    password: str
    email: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str
    model_name: Optional[str] = "openai"
    api_key: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    model_name: Optional[str] = "openai"
    api_key: Optional[str] = None
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    message_id: str
    timestamp: str
    model_used: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str

# LLM Handler
async def get_llm_response(message: str, model_name: str, api_key: str) -> str:
    """Get response from LLM provider"""
    
    # For demo, return Yappy personality response
    # In production, integrate real LLMs here
    
    if not api_key:
        return "Woof! 🐕 Please provide an API key to use AI features!"
    
    # Demo responses based on model
    if model_name == "openai":
        return f"Woof! 🐕 [OpenAI Mode] You said: '{message}'. I'm Yappy, your friendly AI assistant! In production, this would use GPT-4."
    elif model_name == "anthropic":
        return f"Woof! 🐕 [Claude Mode] Tail-waggingly happy to help! You said: '{message}'. In production, this would use Claude."
    elif model_name == "google":
        return f"Woof! 🐕 [Gemini Mode] *excited tail wag* You said: '{message}'. In production, this would use Google's Gemini."
    elif model_name == "groq":
        return f"Woof! 🐕 [Groq Mode] Super fast response! You said: '{message}'. In production, this would use Groq's fast inference."
    else:
        return f"Woof! 🐕 You said: '{message}'. I'm ready to help!"

# Helper functions
def create_token(username: str) -> str:
    """Create a simple token (use JWT in production)"""
    return f"token_{username}_{uuid.uuid4().hex[:8]}"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify token and return username"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    # In production, verify JWT token
    # For now, just extract username
    token_parts = credentials.credentials.split("_")
    if len(token_parts) >= 2:
        return token_parts[1]
    raise HTTPException(status_code=401, detail="Invalid token")

# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a simple frontend"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Yappy AI Assistant</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: #f5f5f5;
            }
            .header {
                text-align: center;
                margin-bottom: 40px;
            }
            .yappy-logo {
                font-size: 80px;
                animation: bounce 2s infinite;
            }
            @keyframes bounce {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-20px); }
            }
            .chat-container {
                background: white;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .api-info {
                background: #e3f2fd;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
            }
            code {
                background: #f5f5f5;
                padding: 2px 6px;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
            }
            .endpoint {
                margin: 10px 0;
                padding: 10px;
                background: #fafafa;
                border-radius: 6px;
            }
            .method {
                display: inline-block;
                padding: 2px 8px;
                border-radius: 4px;
                font-weight: bold;
                margin-right: 10px;
            }
            .method-get { background: #4caf50; color: white; }
            .method-post { background: #2196f3; color: white; }
            .button {
                background: #2196f3;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 16px;
            }
            .button:hover {
                background: #1976d2;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="yappy-logo">🐕</div>
            <h1>Yappy AI Assistant</h1>
            <p>Your friendly AI companion powered by multiple LLMs</p>
        </div>
        
        <div class="api-info">
            <h2>🚀 API Endpoints</h2>
            
            <div class="endpoint">
                <span class="method method-post">POST</span>
                <code>/api/signup</code> - Create a new account
            </div>
            
            <div class="endpoint">
                <span class="method method-post">POST</span>
                <code>/api/login</code> - Login and get access token
            </div>
            
            <div class="endpoint">
                <span class="method method-post">POST</span>
                <code>/api/chat</code> - Chat with Yappy (requires auth)
            </div>
            
            <div class="endpoint">
                <span class="method method-get">GET</span>
                <code>/api/conversations</code> - Get your conversations (requires auth)
            </div>
            
            <div class="endpoint">
                <span class="method method-get">GET</span>
                <code>/api/models</code> - List available AI models
            </div>
        </div>
        
        <div class="chat-container">
            <h2>🤖 Supported Models</h2>
            <ul>
                <li><strong>OpenAI</strong> - GPT-3.5 & GPT-4</li>
                <li><strong>Anthropic</strong> - Claude 3</li>
                <li><strong>Google</strong> - Gemini Pro</li>
                <li><strong>Groq</strong> - Fast open models</li>
            </ul>
            
            <h2>📚 Interactive Documentation</h2>
            <p>Explore and test all endpoints:</p>
            <a href="/docs" target="_blank">
                <button class="button">Open API Documentation</button>
            </a>
        </div>
        
        <div class="chat-container">
            <h2>🔧 Quick Start</h2>
            <p>1. Sign up for an account</p>
            <p>2. Login with your API key</p>
            <p>3. Start chatting with Yappy!</p>
            
            <h3>Example Request:</h3>
            <pre style="background: #f5f5f5; padding: 15px; border-radius: 6px; overflow-x: auto;">
curl -X POST https://yappyai.onrender.com/api/chat \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{
    "message": "Hello Yappy!",
    "model_name": "openai",
    "api_key": "your-api-key"
  }'</pre>
        </div>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

@app.post("/api/signup", response_model=TokenResponse)
async def signup(user: UserSignup):
    """Create a new user account"""
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Store user (hash password in production)
    users_db[user.username] = {
        "password": user.password,
        "email": user.email,
        "created_at": datetime.now().isoformat(),
        "conversations": []
    }
    
    # Create token
    token = create_token(user.username)
    
    return TokenResponse(
        access_token=token,
        username=user.username
    )

@app.post("/api/login", response_model=TokenResponse)
async def login(user: UserLogin):
    """Login and get access token"""
    if user.username not in users_db:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if users_db[user.username]["password"] != user.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Store API key if provided
    if user.api_key:
        api_keys_db[user.username] = {
            user.model_name: user.api_key
        }
    
    # Create token
    token = create_token(user.username)
    
    return TokenResponse(
        access_token=token,
        username=user.username
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    username: str = Depends(verify_token)
):
    """Chat with Yappy AI"""
    
    # Get or create conversation
    conv_id = request.conversation_id or str(uuid.uuid4())
    if conv_id not in conversations_db:
        conversations_db[conv_id] = {
            "user": username,
            "messages": [],
            "created_at": datetime.now().isoformat()
        }
    
    # Verify conversation belongs to user
    if conversations_db[conv_id]["user"] != username:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Get API key
    api_key = request.api_key
    if not api_key and username in api_keys_db:
        api_key = api_keys_db[username].get(request.model_name)
    
    # Get LLM response
    response_text = await get_llm_response(
        request.message,
        request.model_name,
        api_key
    )
    
    # Store message
    message_id = str(uuid.uuid4())
    message_data = {
        "id": message_id,
        "user_message": request.message,
        "assistant_response": response_text,
        "model": request.model_name,
        "timestamp": datetime.now().isoformat()
    }
    conversations_db[conv_id]["messages"].append(message_data)
    
    return ChatResponse(
        response=response_text,
        conversation_id=conv_id,
        message_id=message_id,
        timestamp=message_data["timestamp"],
        model_used=request.model_name
    )

@app.get("/api/conversations")
async def get_conversations(username: str = Depends(verify_token)):
    """Get user's conversations"""
    user_conversations = []
    
    for conv_id, conv_data in conversations_db.items():
        if conv_data["user"] == username:
            user_conversations.append({
                "id": conv_id,
                "created_at": conv_data["created_at"],
                "message_count": len(conv_data["messages"]),
                "last_message": conv_data["messages"][-1]["timestamp"] if conv_data["messages"] else None
            })
    
    return {"conversations": user_conversations}

@app.get("/api/models")
async def get_models():
    """Get available AI models"""
    return {
        "models": [
            {"id": "openai", "name": "OpenAI GPT", "description": "GPT-3.5 and GPT-4"},
            {"id": "anthropic", "name": "Anthropic Claude", "description": "Claude 3 models"},
            {"id": "google", "name": "Google Gemini", "description": "Gemini Pro"},
            {"id": "groq", "name": "Groq", "description": "Fast open-source models"}
        ]
    }

@app.get("/api/profile")
async def get_profile(username: str = Depends(verify_token)):
    """Get user profile"""
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = users_db[username]
    conv_count = sum(1 for c in conversations_db.values() if c["user"] == username)
    
    return {
        "username": username,
        "email": user.get("email"),
        "created_at": user.get("created_at"),
        "conversation_count": conv_count
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🐕 Woof! Yappy AI v2.0 starting on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)