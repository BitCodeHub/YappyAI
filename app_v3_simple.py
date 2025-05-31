import os
import json
import uuid
import hashlib
import secrets
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
from databases import Database
import sqlalchemy
from sqlalchemy import MetaData, Table, Column, Integer, String, DateTime, Text, JSON
import httpx
import re

# Import LLM libraries
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from groq import Groq
except ImportError:
    Groq = None

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./yappy_v3.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

database = Database(DATABASE_URL)
metadata = MetaData()

# Database Tables
users_table = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String(50), unique=True, index=True, nullable=False),
    Column("email", String(100)),
    Column("password_hash", String(255), nullable=False),
    Column("api_keys", JSON, default={}),
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
    Column("updated_at", DateTime, default=datetime.now),
)

engine = sqlalchemy.create_engine(DATABASE_URL)

# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🐕 Yappy AI v3 Simple - Starting up...")
    try:
        await database.connect()
        metadata.create_all(bind=engine)
        print("✅ Database connected")
    except Exception as e:
        print(f"❌ Database error: {e}")
    yield
    await database.disconnect()

# Initialize FastAPI
app = FastAPI(
    title="Yappy AI v3 Simple",
    description="AI with working web search",
    version="3.1.0",
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

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Security
security = HTTPBearer(auto_error=False)

# Models
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

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    message_id: str
    timestamp: str
    model_used: str
    web_searched: bool = False
    search_query: Optional[str] = None

# Helpers
def hash_password(password: str) -> str:
    return hashlib.sha256(f"{password}yappy2024".encode()).hexdigest()

def create_token(username: str) -> str:
    return f"{username}:{secrets.token_urlsafe(32)}"

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        username = credentials.credentials.split(":")[0]
        return username
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

# Simple Web Search using a working API
class SimpleWebSearch:
    @staticmethod
    async def search(query: str) -> str:
        """Simple web search that actually works"""
        try:
            # Use DuckDuckGo's API
            async with httpx.AsyncClient() as client:
                # Try to get instant answer first
                instant_response = await client.get(
                    "https://api.duckduckgo.com/",
                    params={
                        "q": query,
                        "format": "json",
                        "no_html": "1"
                    },
                    timeout=5.0
                )
                
                search_info = ""
                
                if instant_response.status_code == 200:
                    data = instant_response.json()
                    
                    # Get instant answer
                    if data.get("Answer"):
                        search_info += f"Answer: {data['Answer']}\n\n"
                    
                    # Get abstract
                    if data.get("Abstract"):
                        search_info += f"Summary: {data['Abstract']}\n"
                        if data.get("AbstractSource"):
                            search_info += f"Source: {data['AbstractSource']}\n\n"
                    
                    # Get definition
                    if data.get("Definition"):
                        search_info += f"Definition: {data['Definition']}\n"
                        if data.get("DefinitionSource"):
                            search_info += f"Source: {data['DefinitionSource']}\n\n"
                
                # If no instant answer, try web search
                if not search_info:
                    # Try a simple web scraping approach
                    search_url = f"https://html.duckduckgo.com/html/?q={query}"
                    web_response = await client.get(
                        search_url,
                        headers={"User-Agent": "Mozilla/5.0"},
                        timeout=5.0
                    )
                    
                    if web_response.status_code == 200:
                        # Extract snippets from HTML
                        text = web_response.text
                        snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', text)
                        
                        if snippets:
                            search_info = "Web Search Results:\n\n"
                            for i, snippet in enumerate(snippets[:3], 1):
                                clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                                if clean_snippet:
                                    search_info += f"{i}. {clean_snippet}\n\n"
                
                # For NBA/sports queries, add specific search
                if any(sport in query.lower() for sport in ['nba', 'basketball', 'finals', 'game', 'score', 'won']):
                    # Try to get sports scores
                    sports_response = await client.get(
                        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
                        timeout=3.0
                    )
                    
                    if sports_response.status_code == 200:
                        sports_data = sports_response.json()
                        search_info = "NBA Latest Scores and Results:\n\n"
                        
                        # Get recent games
                        events = sports_data.get("events", [])
                        for event in events[:5]:
                            name = event.get("name", "")
                            status = event.get("status", {}).get("type", {}).get("description", "")
                            competitions = event.get("competitions", [{}])[0]
                            
                            # Get teams and scores
                            competitors = competitions.get("competitors", [])
                            if len(competitors) >= 2:
                                team1 = competitors[0]
                                team2 = competitors[1]
                                
                                team1_name = team1.get("team", {}).get("displayName", "Team 1")
                                team1_score = team1.get("score", "0")
                                team2_name = team2.get("team", {}).get("displayName", "Team 2")
                                team2_score = team2.get("score", "0")
                                
                                search_info += f"{name}\n"
                                search_info += f"Status: {status}\n"
                                search_info += f"{team1_name}: {team1_score} - {team2_name}: {team2_score}\n\n"
                
                # Special handling for specific queries
                if "weather" in query.lower():
                    # Extract location
                    location = "New York"  # Default
                    location_match = re.search(r'weather (?:in|at|for) ([\w\s,]+)', query, re.IGNORECASE)
                    if location_match:
                        location = location_match.group(1).strip()
                    
                    # Get weather
                    weather_response = await client.get(
                        f"https://wttr.in/{location}?format=3",
                        timeout=3.0
                    )
                    
                    if weather_response.status_code == 200:
                        weather_text = weather_response.text.strip()
                        search_info = f"Current Weather: {weather_text}\n\n"
                        
                        # Get detailed weather
                        detailed_response = await client.get(
                            f"https://wttr.in/{location}?format=j1",
                            timeout=3.0
                        )
                        
                        if detailed_response.status_code == 200:
                            weather_data = detailed_response.json()
                            current = weather_data.get("current_condition", [{}])[0]
                            search_info += f"Temperature: {current.get('temp_F', 'N/A')}°F ({current.get('temp_C', 'N/A')}°C)\n"
                            search_info += f"Feels like: {current.get('FeelsLikeF', 'N/A')}°F\n"
                            search_info += f"Condition: {current.get('weatherDesc', [{}])[0].get('value', 'N/A')}\n"
                            search_info += f"Humidity: {current.get('humidity', 'N/A')}%\n"
                            search_info += f"Wind: {current.get('windspeedMiles', 'N/A')} mph"
                
                return search_info if search_info else "No specific search results found, but I'll do my best to answer based on general knowledge."
                
        except Exception as e:
            print(f"Search error: {e}")
            return f"Search encountered an error, but I'll still try to help with your question."

# LLM Handler
class LLMHandler:
    def __init__(self):
        self.system_prompt = """You are Yappy, a helpful AI assistant with web search capabilities.
        
        When you receive search results, use them to provide current, accurate information.
        Always indicate when information comes from search results vs. your general knowledge.
        Be friendly and helpful while being accurate."""
    
    async def get_response(self, message: str, search_info: str, model_name: str, api_key: str) -> Tuple[str, int]:
        """Get response with search context"""
        
        # Create enhanced message with search results
        if search_info and search_info != "No specific search results found, but I'll do my best to answer based on general knowledge.":
            enhanced_message = f"""I searched for: "{message}"

Here's what I found:

{search_info}

Based on these search results, please answer: {message}

Important: Use the search results above to provide current, accurate information."""
        else:
            enhanced_message = message
        
        if not api_key and model_name != "demo":
            return "Woof! 🐕 I need an API key to provide AI responses. Please add one in your profile!", 0
        
        try:
            # OpenAI
            if model_name == "openai" and openai and api_key:
                client = openai.OpenAI(api_key=api_key)
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": enhanced_message}
                    ],
                    max_tokens=800,
                    temperature=0.7
                )
                
                return response.choices[0].message.content, response.usage.total_tokens
            
            # Anthropic
            elif model_name == "anthropic" and anthropic and api_key:
                client = anthropic.Anthropic(api_key=api_key)
                
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=800,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": enhanced_message}]
                )
                
                return response.content[0].text, response.usage.input_tokens + response.usage.output_tokens
            
            # Google
            elif model_name == "google" and api_key:
                import requests
                
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
                
                response = requests.post(url, json={
                    "contents": [{
                        "parts": [{"text": f"{self.system_prompt}\n\n{enhanced_message}"}]
                    }]
                })
                
                if response.status_code == 200:
                    data = response.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"], 100
                
            # Groq
            elif model_name == "groq" and Groq and api_key:
                client = Groq(api_key=api_key)
                
                response = client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": enhanced_message}
                    ],
                    max_tokens=800
                )
                
                return response.choices[0].message.content, response.usage.total_tokens
            
            # Demo
            else:
                if search_info and "Weather" in search_info:
                    return f"Woof! 🐕 Here's what I found:\n\n{search_info}\n\nFor full AI features, add an API key!", 50
                else:
                    return "Woof! 🐕 I'm in demo mode. Add an API key for full features with web search!", 50
                
        except Exception as e:
            return f"Error: {str(e)}", 0

# API Endpoints

@app.get("/")
async def root():
    return RedirectResponse(url="/static/yappy_v3.html")

@app.post("/auth/register")
async def register(user_data: UserSignup):
    try:
        # Check if exists
        query = users_table.select().where(users_table.c.username == user_data.username)
        existing = await database.fetch_one(query)
        
        if existing:
            raise HTTPException(status_code=400, detail="Username exists")
        
        # Create user
        query = users_table.insert().values(
            username=user_data.username,
            email=user_data.email,
            password_hash=hash_password(user_data.password),
            created_at=datetime.now(),
            api_keys={}
        )
        
        await database.execute(query)
        
        return {
            "access_token": create_token(user_data.username),
            "token_type": "bearer",
            "username": user_data.username
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/login")
async def login(user_data: UserLogin):
    query = users_table.select().where(users_table.c.username == user_data.username)
    user = await database.fetch_one(query)
    
    if not user or user["password_hash"] != hash_password(user_data.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return {
        "access_token": create_token(user_data.username),
        "token_type": "bearer",
        "username": user_data.username
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, username: str = Depends(verify_token)):
    """Chat with web search"""
    try:
        # Get user
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get API key
        api_key = user.get("api_keys", {}).get(request.model_name, "")
        
        # Determine if we need to search
        query_lower = request.message.lower()
        needs_search = False
        
        # Always search for questions
        if '?' in request.message:
            needs_search = True
        
        # Search for specific patterns
        search_patterns = [
            'what', 'who', 'where', 'when', 'how', 'why',
            'latest', 'current', 'today', 'news', 'weather',
            'price', 'score', 'won', 'lost', 'result',
            'happening', 'update', 'recent', '2024', '2025'
        ]
        
        if any(pattern in query_lower for pattern in search_patterns):
            needs_search = True
        
        # Perform search if needed
        search_info = ""
        if needs_search:
            print(f"🔍 Searching for: {request.message}")
            search_info = await SimpleWebSearch.search(request.message)
            print(f"📊 Search completed")
        
        # Get LLM response
        llm_handler = LLMHandler()
        response_text, tokens = await llm_handler.get_response(
            request.message,
            search_info,
            request.model_name,
            api_key
        )
        
        # Create/update conversation
        if not request.conversation_id:
            request.conversation_id = str(uuid.uuid4())
            
            conv_query = conversations_table.insert().values(
                id=request.conversation_id,
                user_id=user["id"],
                title=request.message[:50],
                messages=[],
                created_at=datetime.now()
            )
            await database.execute(conv_query)
        
        # Save message
        conv_query = conversations_table.select().where(
            conversations_table.c.id == request.conversation_id
        )
        conversation = await database.fetch_one(conv_query)
        messages = conversation["messages"] if conversation else []
        
        messages.append({
            "user_message": request.message,
            "assistant_response": response_text,
            "timestamp": datetime.now().isoformat(),
            "web_searched": needs_search,
            "search_info": search_info if needs_search else None
        })
        
        update_query = conversations_table.update().where(
            conversations_table.c.id == request.conversation_id
        ).values(
            messages=messages,
            updated_at=datetime.now()
        )
        await database.execute(update_query)
        
        return ChatResponse(
            response=response_text,
            conversation_id=request.conversation_id,
            message_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            model_used=request.model_name,
            web_searched=needs_search,
            search_query=request.message if needs_search else None
        )
        
    except Exception as e:
        print(f"Chat error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/user/api-key")
async def update_api_key(update_data: UpdateApiKey, username: str = Depends(verify_token)):
    query = users_table.select().where(users_table.c.username == username)
    user = await database.fetch_one(query)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    api_keys = user["api_keys"] or {}
    api_keys[update_data.model_name] = update_data.api_key
    
    update_query = users_table.update().where(
        users_table.c.username == username
    ).values(api_keys=api_keys)
    
    await database.execute(update_query)
    
    return {"status": "success"}

@app.get("/api/conversations")
async def get_conversations(username: str = Depends(verify_token)):
    query = users_table.select().where(users_table.c.username == username)
    user = await database.fetch_one(query)
    
    conv_query = conversations_table.select().where(
        conversations_table.c.user_id == user["id"]
    ).order_by(conversations_table.c.updated_at.desc())
    
    conversations = await database.fetch_all(conv_query)
    
    return [{
        "id": conv["id"],
        "title": conv["title"],
        "created_at": conv["created_at"].isoformat()
    } for conv in conversations]

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "3.1.0"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🐕 Starting Yappy AI v3 Simple on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)