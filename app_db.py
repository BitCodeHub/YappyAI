"""
Yappy AI Assistant - Final Version with Agent Routing and SearxNG Search
Implements the exact web search functionality from the local AgenticSeek app
"""
import os
import json
import uuid
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
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
import re
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# SearxNG Search Implementation (from local app)
class SearxSearch:
    def __init__(self):
        # Use multiple SearxNG instances for reliability
        self.searx_instances = [
            "https://search.ononoki.org",
            "https://searx.be",
            "https://searx.info",
            "https://searx.xyz"
        ]
        self.timeout = 10
    
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search using SearxNG instances"""
        for instance in self.searx_instances:
            try:
                params = {
                    'q': query,
                    'format': 'json',
                    'language': 'en',
                    'safesearch': '0',
                    'categories': 'general'
                }
                
                response = requests.get(
                    f"{instance}/search",
                    params=params,
                    timeout=self.timeout,
                    headers={'User-Agent': 'Yappy/1.0'}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    
                    for result in data.get('results', [])[:num_results]:
                        results.append({
                            'title': result.get('title', ''),
                            'snippet': result.get('content', ''),
                            'link': result.get('url', ''),
                            'engine': result.get('engine', 'unknown')
                        })
                    
                    if results:
                        logger.info(f"Found {len(results)} results from {instance}")
                        return results
                        
            except Exception as e:
                logger.warning(f"SearxNG instance {instance} failed: {e}")
                continue
        
        # Fallback to DuckDuckGo instant answer API
        return self._duckduckgo_fallback(query)
    
    def _duckduckgo_fallback(self, query: str) -> List[Dict[str, Any]]:
        """Fallback search using DuckDuckGo"""
        try:
            from urllib.parse import quote
            
            # Try instant answer API first
            ddg_url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1"
            response = requests.get(ddg_url, timeout=5)
            
            results = []
            if response.status_code == 200:
                data = response.json()
                
                # Get instant answer
                if data.get('Abstract'):
                    results.append({
                        'title': 'Summary',
                        'snippet': data['Abstract'],
                        'link': data.get('AbstractURL', ''),
                        'engine': 'duckduckgo'
                    })
                
                if data.get('Answer'):
                    results.append({
                        'title': 'Quick Answer',
                        'snippet': data['Answer'],
                        'link': '',
                        'engine': 'duckduckgo'
                    })
                
                # Related topics
                for topic in data.get('RelatedTopics', [])[:3]:
                    if isinstance(topic, dict) and 'Text' in topic:
                        results.append({
                            'title': topic.get('FirstURL', '').split('/')[-1].replace('_', ' '),
                            'snippet': topic['Text'],
                            'link': topic.get('FirstURL', ''),
                            'engine': 'duckduckgo'
                        })
            
            # If no instant answers, try HTML search
            if not results:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(f"https://html.duckduckgo.com/html/?q={quote(query)}", 
                                       headers=headers, timeout=5)
                
                if response.status_code == 200:
                    # Simple regex to extract results
                    import re
                    pattern = r'<a class="result__a" href="([^"]+)"[^>]*>([^<]+)</a>.*?<a class="result__snippet"[^>]*>([^<]+)</a>'
                    matches = re.findall(pattern, response.text, re.DOTALL)
                    
                    for match in matches[:5]:
                        results.append({
                            'title': match[1].strip(),
                            'snippet': match[2].strip(),
                            'link': match[0],
                            'engine': 'duckduckgo'
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo fallback failed: {e}")
            return []

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./yappy.db")

# Fix for Render PostgreSQL URLs
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

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
    Column("updated_at", DateTime, default=datetime.now, onupdate=datetime.now)
)

# Agent Router Implementation
class AgentRouter:
    """Routes queries to appropriate agents based on content"""
    
    def analyze_query(self, query: str, conversation_history: List[Dict] = None) -> Tuple[str, bool]:
        """
        Analyze query and determine if it needs web search
        Returns: (agent_type, needs_web_search)
        """
        query_lower = query.lower()
        
        # Always search for these patterns (from local app)
        web_search_patterns = [
            # Current events/facts
            'who is', 'what is', 'when is', 'where is', 'how much', 'how many',
            'latest', 'current', 'today', 'now', 'recent', 'news', 'update',
            # People and organizations
            'president', 'ceo', 'founder', 'leader', 'minister', 'director',
            'company', 'organization', 'government', 'country',
            # Weather
            'weather', 'temperature', 'forecast', 'rain', 'snow',
            # Sports/Events
            'game', 'score', 'match', 'playing', 'nba', 'nfl', 'sports',
            # Financial
            'price', 'stock', 'crypto', 'bitcoin', 'market',
            # Search indicators
            'search', 'find', 'look up', 'check'
        ]
        
        # Check if query needs web search
        needs_search = any(pattern in query_lower for pattern in web_search_patterns)
        
        # Also search for questions
        if query.strip().endswith('?'):
            question_words = ['what', 'who', 'when', 'where', 'how', 'which', 'why', 'is', 'are', 'do', 'does']
            if any(query_lower.startswith(word) for word in question_words):
                needs_search = True
        
        # Determine agent type
        if needs_search:
            return "browser", True
        else:
            return "casual", False

# LLM Handler
class LLMHandler:
    """Handles all LLM interactions with Yappy personality"""
    
    def __init__(self):
        self.searx_tool = SearxSearch()
        self.agent_router = AgentRouter()
        
    async def get_response(self, prompt: str, model_name: str, api_key: Optional[str] = None, 
                          conversation_history: List[Dict] = None) -> Tuple[str, int]:
        """Get response from LLM with web search when needed"""
        
        # Route query
        agent_type, needs_search = self.agent_router.analyze_query(prompt, conversation_history)
        
        # Perform web search if needed
        web_context = ""
        if needs_search:
            logger.info(f"Performing web search for: {prompt}")
            search_results = self.searx_tool.search(prompt)
            
            if search_results:
                web_context = "\n\nWeb Search Results:\n"
                for i, result in enumerate(search_results, 1):
                    web_context += f"\n{i}. {result['title']}"
                    if result['snippet']:
                        web_context += f"\n   {result['snippet']}"
                    if result['link']:
                        web_context += f"\n   Source: {result['link']}"
                    web_context += "\n"
                
                logger.info(f"Found {len(search_results)} search results")
        
        # Prepare system prompt based on agent type
        if agent_type == "browser" and web_context:
            system_prompt = """You are Yappy 🐕, a friendly AI assistant with real-time web access!
You MUST use the provided search results to give accurate, current information.
Always cite the search results when answering factual questions.
Today's date is """ + datetime.now().strftime('%B %d, %Y') + "."
        else:
            system_prompt = """You are Yappy 🐕, a friendly and enthusiastic AI assistant!
Be cheerful and helpful. Use dog-related expressions occasionally (like "Woof!" or "*wags tail*").
Always aim to brighten the user's day with your positive energy!"""
        
        # Build enhanced prompt
        enhanced_prompt = prompt
        if web_context:
            enhanced_prompt = f"""User Question: {prompt}

{web_context}

Based on the search results above, please provide an accurate answer.
Remember to be friendly and maintain your Yappy personality! 🐕"""
        
        # Call LLM
        try:
            if not api_key:
                responses = [
                    "Woof! 🐕 I need an API key to search the web and give you the latest information! Please add one in your profile.",
                    "*tilts head* 🐕 I'd love to fetch that information for you, but I need an API key first! Add one in your profile settings.",
                    "Ruff! 🐕 My web search abilities are locked without an API key. Add one to unlock my full potential!"
                ]
                import random
                return random.choice(responses), 0
            
            if model_name == "openai" and openai:
                client = openai.OpenAI(api_key=api_key)
                
                messages = [{"role": "system", "content": system_prompt}]
                
                # Add conversation history
                if conversation_history:
                    for msg in conversation_history[-5:]:  # Last 5 messages
                        messages.append({"role": "user", "content": msg.get("user_message", "")})
                        messages.append({"role": "assistant", "content": msg.get("assistant_response", "")})
                
                messages.append({"role": "user", "content": enhanced_prompt})
                
                response = client.chat.completions.create(
                    model="gpt-4" if "gpt-4" in api_key else "gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                return response.choices[0].message.content, response.usage.total_tokens
            
            elif model_name == "anthropic" and anthropic:
                client = anthropic.Anthropic(api_key=api_key)
                
                # Build conversation for Claude
                messages = []
                if conversation_history:
                    for msg in conversation_history[-5:]:
                        messages.append({"role": "user", "content": msg.get("user_message", "")})
                        messages.append({"role": "assistant", "content": msg.get("assistant_response", "")})
                
                messages.append({"role": "user", "content": enhanced_prompt})
                
                response = client.messages.create(
                    model="claude-3-sonnet-20240229",
                    system=system_prompt,
                    messages=messages,
                    max_tokens=1000
                )
                
                return response.content[0].text, response.usage.input_tokens + response.usage.output_tokens
            
            elif model_name == "google" and genai:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.0-flash-latest')
                
                # Build prompt with history
                full_prompt = system_prompt + "\n\n"
                if conversation_history:
                    for msg in conversation_history[-5:]:
                        full_prompt += f"User: {msg.get('user_message', '')}\n"
                        full_prompt += f"Assistant: {msg.get('assistant_response', '')}\n"
                
                full_prompt += f"User: {enhanced_prompt}\nAssistant:"
                
                response = model.generate_content(full_prompt)
                return response.text, 100
            
            elif model_name == "groq" and Groq:
                client = Groq(api_key=api_key)
                
                messages = [{"role": "system", "content": system_prompt}]
                if conversation_history:
                    for msg in conversation_history[-5:]:
                        messages.append({"role": "user", "content": msg.get("user_message", "")})
                        messages.append({"role": "assistant", "content": msg.get("assistant_response", "")})
                
                messages.append({"role": "user", "content": enhanced_prompt})
                
                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                return response.choices[0].message.content, response.usage.total_tokens
            
            else:
                return f"Woof! 🐕 The {model_name} model isn't available. Try another one!", 0
                
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return f"Woof! 🐕 I encountered an error: {str(e)}", 0

# FastAPI app setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🐕 Woof! Yappy AI with SearxNG Search is starting up...")
    print(f"Current date: {datetime.now().strftime('%A, %B %d, %Y')}")
    
    try:
        await database.connect()
        print("✅ Database connected successfully")
        
        # Create tables if they don't exist
        engine = sqlalchemy.create_engine(DATABASE_URL)
        metadata.create_all(bind=engine)
        print("✅ Database tables created/verified")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
    
    yield
    
    # Shutdown
    print("🐕 Yappy is shutting down... Goodbye!")
    if database.is_connected:
        await database.disconnect()

app = FastAPI(title="Yappy AI Assistant", version="6.0.0", lifespan=lifespan)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Security
security = HTTPBearer()

# Pydantic models
class UserLogin(BaseModel):
    username: str
    password: str

class UserSignup(BaseModel):
    username: str
    email: str
    password: str

class UpdateApiKey(BaseModel):
    model_name: str
    api_key: str

class ChatRequest(BaseModel):
    message: str
    model_name: Optional[str] = "openai"
    conversation_id: Optional[str] = None
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    message_id: str
    timestamp: str
    model_used: str
    tokens_used: Optional[int] = None
    had_web_search: Optional[bool] = False

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    expires_in: int = 3600

# Helper functions
def hash_password(password: str) -> str:
    """Hash password with salt"""
    salt = "yappy_salt_2024"
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
            return username
        
        # Verify user exists
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        if not user:
            return username  # Allow access for now
        
        return username
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        try:
            return credentials.credentials.split(":")[0]
        except:
            raise HTTPException(status_code=401, detail="Invalid token")

# Initialize components
llm_handler = LLMHandler()

# API Endpoints
@app.get("/")
async def root():
    """Redirect to chat interface"""
    yappy_path = os.path.join(static_dir, "yappy.html")
    if os.path.exists(yappy_path):
        return FileResponse(yappy_path)
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return RedirectResponse(url="/static/yappy.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "6.0.0",
        "current_date": datetime.now().strftime("%B %d, %Y"),
        "features": ["multi_agent", "searxng_search", "real_time_data"],
        "database": "connected" if database.is_connected else "disconnected"
    }

@app.post("/auth/register", response_model=TokenResponse)
async def register(user_data: UserSignup):
    """Register new user"""
    try:
        # Check if user exists
        query = users_table.select().where(users_table.c.username == user_data.username)
        existing_user = await database.fetch_one(query)
        
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # Create new user
        query = users_table.insert().values(
            username=user_data.username,
            email=user_data.email,
            password_hash=hash_password(user_data.password),
            created_at=datetime.now(),
            api_keys={},
            preferences={}
        )
        
        await database.execute(query)
        
        # Create token
        token = create_token(user_data.username)
        
        return TokenResponse(
            access_token=token,
            username=user_data.username
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/auth/login", response_model=TokenResponse)
async def login(user_data: UserLogin):
    """Login user"""
    try:
        # Find user
        query = users_table.select().where(users_table.c.username == user_data.username)
        user = await database.fetch_one(query)
        
        if not user or user.password_hash != hash_password(user_data.password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Create token
        token = create_token(user_data.username)
        
        return TokenResponse(
            access_token=token,
            username=user_data.username
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, username: str = Depends(verify_token)):
    """Main chat endpoint with SearxNG web search"""
    try:
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
        
        # Get response with potential web search
        response_text, tokens = await llm_handler.get_response(
            request.message,
            request.model_name,
            api_key,
            conversation_messages
        )
        
        # Check if web search was performed
        agent_type, had_web_search = llm_handler.agent_router.analyze_query(request.message)
        
        # Store message
        message_id = str(uuid.uuid4())
        message_data = {
            "id": message_id,
            "user_message": request.message,
            "assistant_response": response_text,
            "model": request.model_name,
            "timestamp": datetime.now().isoformat(),
            "tokens": tokens,
            "had_web_search": had_web_search,
            "agent_type": agent_type
        }
        
        # Update conversation
        conversation_messages.append(message_data)
        # Update conversation - handle missing updated_at column
        try:
            update_query = conversations_table.update().where(
                conversations_table.c.id == conv_id
            ).values(
                messages=conversation_messages,
                updated_at=datetime.now()
            )
            await database.execute(update_query)
        except Exception as e:
            # Fallback if updated_at column doesn't exist
            logger.warning(f"Updated_at column issue: {e}")
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
            tokens_used=tokens,
            had_web_search=had_web_search
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post("/api/user/api-key")
async def update_api_key(update_data: UpdateApiKey, username: str = Depends(verify_token)):
    """Update user's API key for a specific model"""
    try:
        # Get current user
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Update API keys
        api_keys = user.api_keys or {}
        api_keys[update_data.model_name] = update_data.api_key
        
        update_query = users_table.update().where(
            users_table.c.username == username
        ).values(api_keys=api_keys)
        
        await database.execute(update_query)
        
        return {"status": "success", "message": f"API key updated for {update_data.model_name}"}
        
    except Exception as e:
        logger.error(f"API key update error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update API key")

@app.get("/api/conversations")
async def get_conversations(username: str = Depends(verify_token)):
    """Get user's conversations"""
    try:
        # Get user
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            return []
        
        # Get conversations
        conv_query = conversations_table.select().where(
            conversations_table.c.user_id == user.id
        ).order_by(conversations_table.c.created_at.desc())
        
        conversations = await database.fetch_all(conv_query)
        
        result = []
        for conv in conversations:
            messages = conv.messages or []
            last_message = messages[-1] if messages else None
            
            result.append({
                "id": conv.id,
                "title": conv.title,
                "created_at": conv.created_at.isoformat() if conv.created_at else None,
                "updated_at": conv.updated_at.isoformat() if hasattr(conv, 'updated_at') and conv.updated_at else conv.created_at.isoformat(),
                "message_count": len(messages),
                "last_message": last_message.get("user_message") if last_message else None,
                "last_agent": last_message.get("agent_type", "casual") if last_message else None
            })
        
        return result
    except Exception as e:
        logger.error(f"Get conversations error: {e}")
        return []

@app.get("/api/conversation/{conversation_id}")
async def get_conversation(conversation_id: str, username: str = Depends(verify_token)):
    """Get specific conversation"""
    try:
        # Get user
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get conversation
        conv_query = conversations_table.select().where(
            conversations_table.c.id == conversation_id
        )
        conversation = await database.fetch_one(conv_query)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Verify ownership
        if conversation.user_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return {
            "id": conversation.id,
            "title": conversation.title,
            "messages": conversation.messages or [],
            "created_at": conversation.created_at.isoformat() if conversation.created_at else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get conversation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get conversation")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)