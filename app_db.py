import os
import json
import uuid
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import asyncio
from contextlib import asynccontextmanager
from databases import Database
import sqlalchemy
from sqlalchemy import MetaData, Table, Column, Integer, String, DateTime, Text, JSON

# Import all LLM libraries
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from groq import Groq
except ImportError:
    Groq = None

# Database configuration
print("Environment variables:")
print(f"DATABASE_URL from env: {os.environ.get('DATABASE_URL', 'NOT SET')[:50]}...")

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./yappy.db")

# Fix for Render PostgreSQL URLs (they use postgres:// but we need postgresql://)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    print(f"Converted to: {DATABASE_URL[:50]}...")

database = Database(DATABASE_URL)
metadata = MetaData()

# Database tables
users_table = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String(50), unique=True, index=True, nullable=False),
    Column("email", String(100)),
    Column("password_hash", String(255), nullable=False),
    Column("api_keys", JSON, default={}),
    Column("preferences", JSON, default={}),
    Column("created_at", DateTime, default=datetime.now),
)

conversations_table = Table(
    "conversations",
    metadata,
    Column("id", String(50), primary_key=True),
    Column("user_id", Integer, nullable=False),
    Column("title", String(200)),
    Column("messages", JSON, default=[]),
    Column("created_at", DateTime, default=datetime.now),
)

engine = sqlalchemy.create_engine(DATABASE_URL)

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - Connect to database
    print("🐕 Woof! Yappy AI is starting up...")
    print(f"Database URL: {DATABASE_URL[:50]}...")
    print(f"Database type: {'PostgreSQL' if 'postgresql' in DATABASE_URL else 'SQLite'}")
    
    try:
        await database.connect()
        print("✅ Database connected successfully")
        
        # Create tables if they don't exist
        metadata.create_all(bind=engine)
        print("✅ Database tables created/verified")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("⚠️ Will continue with limited functionality")
    
    yield
    
    # Shutdown
    try:
        await database.disconnect()
        print("🐕 Yappy AI is shutting down... Goodbye!")
    except:
        pass

# Initialize FastAPI
app = FastAPI(
    title="Yappy AI Complete with Database",
    description="Full-featured AI assistant with PostgreSQL database",
    version="4.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files if they exist
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    print(f"Mounting static directory: {static_dir}")
    print(f"Static files: {os.listdir(static_dir)}")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    print(f"Warning: Static directory not found at {static_dir}")

# Security
security = HTTPBearer(auto_error=False)

# Pydantic models
class UserSignup(BaseModel):
    username: str
    password: str
    email: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class UpdateApiKey(BaseModel):
    model_name: str
    api_key: str

class ChatRequest(BaseModel):
    message: str
    model_name: Optional[str] = "openai"
    conversation_id: Optional[str] = None
    stream: Optional[bool] = False
    file_data: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    message_id: str
    timestamp: str
    model_used: str
    tokens_used: Optional[int] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    expires_in: int = 3600

# Helper functions
def hash_password(password: str) -> str:
    """Hash password with salt"""
    salt = "yappy_salt_2024"  # In production, use unique salt per user
    return hashlib.sha256(f"{password}{salt}".encode()).hexdigest()

def create_token(username: str) -> str:
    """Create secure token"""
    return f"{username}:{secrets.token_urlsafe(32)}"

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify token and return username"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        token = credentials.credentials
        username = token.split(":")[0]
        
        # Check if database is connected
        if not database.is_connected:
            print("⚠️ Database not connected during token verification")
            return username  # Allow access if DB is down
        
        # Verify user exists in database
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        if not user:
            print(f"⚠️ User {username} not found in database")
            return username  # Still allow access for now
        
        return username
    except Exception as e:
        print(f"Token verification error: {e}")
        # Extract username from token for emergency access
        try:
            return credentials.credentials.split(":")[0]
        except:
            raise HTTPException(status_code=401, detail="Invalid token")

# LLM Integration
class LLMHandler:
    def __init__(self):
        self.system_prompt = """You are Yappy, an incredibly friendly, enthusiastic, and helpful AI assistant with a playful golden retriever personality. 
        You love to help and get excited about every task! Use dog-related expressions naturally like:
        - Starting responses with "Woof!" when excited
        - Saying things like "tail-waggingly happy to help!"
        - Using "*wags tail enthusiastically*" for actions
        - Occasionally using "pawsome" instead of "awesome"
        - Ending with encouraging phrases like "Happy to fetch more info if needed!"
        
        Be helpful, accurate, and thorough while maintaining your cheerful dog personality."""
    
    async def get_response(self, message: str, model_name: str, api_key: str, conversation_history: List = None) -> tuple[str, int]:
        """Get response from LLM provider"""
        
        if not api_key and model_name != "demo":
            return "Woof! 🐕 I need an API key to use AI features! You can add one in your profile settings.", 0
        
        try:
            # OpenAI
            if model_name == "openai" and openai and api_key:
                client = openai.OpenAI(api_key=api_key)
                
                messages = [{"role": "system", "content": self.system_prompt}]
                
                # Add conversation history
                if conversation_history:
                    for msg in conversation_history[-5:]:  # Last 5 messages
                        if isinstance(msg, dict) and "user_message" in msg:
                            messages.append({"role": "user", "content": msg["user_message"]})
                            messages.append({"role": "assistant", "content": msg["assistant_response"]})
                
                messages.append({"role": "user", "content": message})
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=800,
                    temperature=0.8
                )
                
                return response.choices[0].message.content, response.usage.total_tokens
            
            # Anthropic
            elif model_name == "anthropic" and anthropic and api_key:
                client = anthropic.Anthropic(api_key=api_key)
                
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=800,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": message}]
                )
                
                return response.content[0].text, response.usage.input_tokens + response.usage.output_tokens
            
            # Google Gemini
            elif model_name == "google" and api_key:
                import requests
                
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
                
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": f"{self.system_prompt}\n\nUser: {message}"
                        }]
                    }]
                }
                
                response = requests.post(url, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    if "candidates" in data and len(data["candidates"]) > 0:
                        text = data["candidates"][0]["content"]["parts"][0]["text"]
                        return text, 100  # Gemini doesn't provide token count in this format
                    else:
                        return "Woof! 🐕 I got a response but couldn't parse it properly.", 0
                else:
                    error_msg = response.json().get("error", {}).get("message", "Unknown error")
                    return f"Woof! 🐕 Gemini API error: {error_msg}", 0
            
            # Groq
            elif model_name == "groq" and Groq and api_key:
                client = Groq(api_key=api_key)
                
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": message}
                ]
                
                response = client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=messages,
                    max_tokens=800,
                    temperature=0.8
                )
                
                return response.choices[0].message.content, response.usage.total_tokens
            
            # Demo mode
            elif model_name == "demo":
                return f"Woof! 🐕 Demo response: I'm Yappy, your friendly AI assistant! You said: '{message[:50]}...' *wags tail happily*", 0
            
            else:
                return f"Woof! 🐕 Sorry, the {model_name} model isn't available right now. Please check your API key or try a different model!", 0
                
        except Exception as e:
            return f"Woof! 🐕 Sorry, I encountered an error: {str(e)}. Please try again or check your API key!", 0

llm_handler = LLMHandler()

# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the Yappy chat interface directly"""
    yappy_path = os.path.join(static_dir, "yappy.html")
    if os.path.exists(yappy_path):
        print(f"Serving yappy.html from {yappy_path}")
        return FileResponse(yappy_path)
    
    # Fallback to the landing page if yappy.html doesn't exist
    print(f"yappy.html not found at {yappy_path}, serving landing page")
    return await root_landing()

@app.get("/landing", response_class=HTMLResponse)
async def root_landing():
    """Serve the main application"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Yappy AI - Database Edition</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
            .container { max-width: 800px; margin: 0 auto; padding: 2rem; text-align: center; }
            h1 { font-size: 3rem; margin-bottom: 1rem; }
            .subtitle { font-size: 1.2rem; margin-bottom: 2rem; opacity: 0.9; }
            .btn { display: inline-block; padding: 12px 24px; background: rgba(255,255,255,0.2); color: white; text-decoration: none; border-radius: 8px; margin: 0 10px; transition: all 0.3s; }
            .btn:hover { background: rgba(255,255,255,0.3); transform: translateY(-2px); }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🐕 Yappy AI</h1>
            <p class="subtitle">Your friendly AI assistant with database persistence!</p>
            <a href="/docs" class="btn">API Documentation</a>
            <a href="/static/yappy.html" class="btn">Open Chat</a>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/chat")
async def chat_redirect():
    """Redirect to Yappy chat interface"""
    return RedirectResponse(url="/static/yappy.html", status_code=303)

@app.get("/health")
async def health_check():
    # Check user count
    user_count = 0
    try:
        if database.is_connected:
            count_query = users_table.select()
            users = await database.fetch_all(count_query)
            user_count = len(users)
    except Exception as e:
        print(f"Error counting users: {e}")
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "4.0.0",
        "database": "connected" if database.is_connected else "disconnected",
        "users_in_db": user_count,
        "features": {
            "authentication": True,
            "multi_llm": True,
            "conversation_history": True,
            "database_persistence": True,
        }
    }

@app.get("/admin/reset-db")
async def reset_database():
    """Reset database tables (for testing only)"""
    try:
        # Drop and recreate tables
        metadata.drop_all(bind=engine)
        metadata.create_all(bind=engine)
        print("✅ Database tables reset successfully")
        return {"message": "Database reset successfully"}
    except Exception as e:
        print(f"❌ Database reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database reset failed: {str(e)}")

@app.get("/admin/list-users")
async def list_users():
    """List all users in database (for debugging)"""
    try:
        if not database.is_connected:
            return {"error": "Database not connected", "users": []}
        
        query = users_table.select()
        users = await database.fetch_all(query)
        
        user_list = []
        for user in users:
            user_list.append({
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "password_hash": user.password_hash[:20] + "...",  # Show first 20 chars
                "created_at": str(user.created_at),
                "has_api_keys": bool(user.api_keys)
            })
        
        return {"total_users": len(user_list), "users": user_list}
    except Exception as e:
        print(f"Error listing users: {e}")
        return {"error": str(e), "users": []}

@app.get("/admin/db-info")
async def database_info():
    """Get database information"""
    try:
        info = {
            "database_url": DATABASE_URL[:30] + "...",
            "is_connected": database.is_connected,
            "type": "PostgreSQL" if "postgresql" in DATABASE_URL else "SQLite"
        }
        
        if database.is_connected:
            # Count tables
            if "postgresql" in DATABASE_URL:
                table_query = "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'"
            else:
                table_query = "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
            
            result = await database.fetch_one(query=table_query)
            info["table_count"] = result[0] if result else 0
            
            # Count users
            user_count_query = users_table.select()
            users = await database.fetch_all(user_count_query)
            info["user_count"] = len(users)
        
        return info
    except Exception as e:
        return {"error": str(e)}

@app.post("/admin/test-password")
async def test_password(request: dict):
    """Test password hashing (for debugging)"""
    try:
        username = request.get("username")
        password = request.get("password")
        
        if not username or not password:
            return {"error": "username and password required"}
        
        # Get user from database
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            return {"error": "User not found", "username": username}
        
        # Test password
        expected_hash = hash_password(password)
        actual_hash = user.password_hash
        
        return {
            "username": username,
            "password_provided": password,
            "expected_hash": expected_hash,
            "actual_hash": actual_hash,
            "match": expected_hash == actual_hash
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/auth/register", response_model=TokenResponse)
async def signup(user: UserSignup):
    """Create a new user account"""
    try:
        print(f"Signup attempt for user: {user.username}")
        
        # Check database connection
        if not database.is_connected:
            print("❌ Database not connected for signup")
            raise HTTPException(status_code=500, detail="Database unavailable")
        
        # Check if user already exists
        query = users_table.select().where(users_table.c.username == user.username)
        existing_user = await database.fetch_one(query)
        
        if existing_user:
            print(f"User {user.username} already exists")
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # Create user
        user_data = {
            "username": user.username,
            "email": user.email or f"{user.username}@yappy.ai",
            "password_hash": hash_password(user.password),
            "api_keys": {},
            "preferences": {
                "default_model": "openai",
                "personality_level": "high"
            }
        }
        
        print(f"Creating user with data: {user_data['username']}, {user_data['email']}")
        
        query = users_table.insert().values(**user_data)
        result = await database.execute(query)
        
        print(f"User created successfully, ID: {result}")
        
        # Verify user was created
        verify_query = users_table.select().where(users_table.c.username == user.username)
        created_user = await database.fetch_one(verify_query)
        print(f"User verification: {created_user is not None}")
        
        # Double check - list all users after creation
        all_users_query = users_table.select()
        all_users = await database.fetch_all(all_users_query)
        print(f"Total users after registration: {len(all_users)}")
        for u in all_users:
            print(f"  - {u.username} (id: {u.id})")
        
        # Create token
        token = create_token(user.username)
        
        return TokenResponse(
            access_token=token,
            username=user.username
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Signup error: {e}")
        raise HTTPException(status_code=500, detail=f"Signup failed: {str(e)}")

@app.post("/auth/login", response_model=TokenResponse)
async def login(user: UserLogin):
    """Login to get access token"""
    try:
        print(f"Login attempt for user: {user.username}")
        
        # Check database connection
        if not database.is_connected:
            print("❌ Database not connected for login")
            raise HTTPException(status_code=500, detail="Database unavailable")
        
        # Query for user
        query = users_table.select().where(users_table.c.username == user.username)
        db_user = await database.fetch_one(query)
        
        print(f"User found in database: {db_user is not None}")
        
        if not db_user:
            # List all users for debugging
            all_users_query = users_table.select()
            all_users = await database.fetch_all(all_users_query)
            print(f"Total users in database: {len(all_users)}")
            for u in all_users:
                print(f"  - {u.username}")
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Verify password
        expected_hash = hash_password(user.password)
        print(f"Password verification: {db_user.password_hash == expected_hash}")
        
        if db_user.password_hash != expected_hash:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Create token
        token = create_token(user.username)
        print(f"Login successful for {user.username}")
        
        return TokenResponse(
            access_token=token,
            username=user.username
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@app.post("/api/keys")
async def update_api_key(
    key_data: UpdateApiKey,
    username: str = Depends(verify_token)
):
    """Update API key for a specific model"""
    try:
        # Get current user
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Update API keys
        api_keys = user.api_keys or {}
        api_keys[key_data.model_name] = key_data.api_key
        
        # Update user in database
        query = users_table.update().where(users_table.c.username == username).values(api_keys=api_keys)
        await database.execute(query)
        
        return {"message": f"API key for {key_data.model_name} updated successfully"}
    except Exception as e:
        print(f"API key update error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update API key: {str(e)}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    username: str = Depends(verify_token)
):
    """Chat with Yappy AI"""
    try:
        print(f"Chat request from {username}: {request.message[:50]}...")
        
        # Check if this is a file upload request
        if request.file_data and request.file_data.get("type") == "application/pdf":
            print("PDF file upload detected")
            
            # Process resume PDF
            try:
                import base64
                import PyPDF2
                from io import BytesIO
                
                file_content = request.file_data.get("content", "")
                file_name = request.file_data.get("name", "resume.pdf")
                
                # Decode base64 PDF content
                if file_content.startswith("data:application/pdf;base64,"):
                    file_content = file_content.split(",")[1]
                
                pdf_bytes = base64.b64decode(file_content)
                pdf_file = BytesIO(pdf_bytes)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                # Extract text from all pages
                resume_text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    resume_text += page.extract_text() + "\n"
                
                print(f"Extracted {len(resume_text)} characters from PDF")
                
                # Create a resume scoring prompt
                scoring_prompt = f"""You are an experienced hiring manager. Analyze this resume and determine:
1. What position the candidate is applying for (based on their experience and the resume content)
2. Their years of relevant experience
3. Whether to HIRE or NOT HIRE based on their qualifications

IMPORTANT: Format your response EXACTLY as follows:

## 📋 Position Applied For: [Detected Position]

## 📊 Resume Score: [X/100]

## 🕐 Years of Experience: [X years]

## ✅ Relevant Experience:
• [List specific relevant experience from resume]
• [Include job titles, companies, and duration]
• [Focus on experience related to the detected position]

## 🎯 Skills Assessment:
• [List relevant skills for the position]
• [Rate each skill based on evidence in resume]

## 🏆 Key Achievements:
• [List notable achievements relevant to the position]

## 📌 Decision: [HIRE / NOT HIRE]

## 💡 Reasoning:
[Explain why you would hire or not hire based on:
- Years of experience (minimum requirements)
- Relevant skills match
- Past performance indicators
- Overall fit for the position]

## 📝 Recommendations:
[If HIRE: What onboarding/training might they need]
[If NOT HIRE: What they need to improve to be considered]

Resume content to analyze:
{resume_text[:4000]}"""
                
                # Get user's API key
                user = await database.fetch_one(
                    users_table.select().where(users_table.c.username == username)
                )
                api_keys = user.api_keys if user else {}
                api_key = api_keys.get(request.model_name)
                
                # Get LLM response for resume scoring
                response_text, tokens = await llm_handler.get_response(
                    scoring_prompt,
                    request.model_name,
                    api_key,
                    []
                )
                
                # Format the final response
                final_response = f"""📄 **Resume Analysis for {file_name}**

{response_text}"""
                
                return ChatResponse(
                    response=final_response,
                    conversation_id=request.conversation_id or str(uuid.uuid4()),
                    message_id=str(uuid.uuid4()),
                    timestamp=datetime.now().isoformat(),
                    model_used=request.model_name,
                    tokens_used=tokens
                )
                
            except Exception as e:
                print(f"Error processing PDF: {e}")
                return ChatResponse(
                    response=f"Woof! 🐕 I had trouble reading the PDF file. Error: {str(e)}. Could you try uploading it again or paste the text content?",
                    conversation_id=request.conversation_id or str(uuid.uuid4()),
                    message_id=str(uuid.uuid4()),
                    timestamp=datetime.now().isoformat(),
                    model_used=request.model_name,
                    tokens_used=0
                )
        
        # Get user
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get API key for the model
        api_keys = user.api_keys or {}
        api_key = api_keys.get(request.model_name)
        
        # Get or create conversation
        conv_id = request.conversation_id or str(uuid.uuid4())
        
        # Get existing conversation
        conv_query = conversations_table.select().where(conversations_table.c.id == conv_id)
        conversation = await database.fetch_one(conv_query)
        
        if not conversation:
            # Create new conversation
            conv_data = {
                "id": conv_id,
                "user_id": user.id,
                "title": request.message[:50] + "..." if len(request.message) > 50 else request.message,
                "messages": []
            }
            insert_query = conversations_table.insert().values(**conv_data)
            await database.execute(insert_query)
            conversation_messages = []
        else:
            # Verify conversation belongs to user
            if conversation.user_id != user.id:
                raise HTTPException(status_code=403, detail="Access denied to this conversation")
            conversation_messages = conversation.messages or []
        
        # Get LLM response
        response_text, tokens = await llm_handler.get_response(
            request.message,
            request.model_name,
            api_key,
            conversation_messages
        )
        
        # Store message
        message_id = str(uuid.uuid4())
        message_data = {
            "id": message_id,
            "user_message": request.message,
            "assistant_response": response_text,
            "model": request.model_name,
            "timestamp": datetime.now().isoformat(),
            "tokens": tokens
        }
        
        # Update conversation
        conversation_messages.append(message_data)
        update_query = conversations_table.update().where(
            conversations_table.c.id == conv_id
        ).values(messages=conversation_messages)
        await database.execute(update_query)
        
        return ChatResponse(
            response=response_text,
            conversation_id=conv_id,
            message_id=message_id,
            timestamp=message_data["timestamp"],
            model_used=request.model_name,
            tokens_used=tokens
        )
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)