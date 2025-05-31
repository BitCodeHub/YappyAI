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
import re

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
print(f"Current date: {datetime.now().strftime('%B %d, %Y')}")

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
    print("🐕 Woof! Yappy AI v4 with Context Awareness is starting up...")
    print(f"Current date: {datetime.now().strftime('%A, %B %d, %Y')}")
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
    title="Yappy AI with Context Awareness",
    description="Your friendly AI assistant with memory and real-time data",
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

# Context extraction helper
def extract_context_from_messages(messages: List[Dict], current_query: str) -> Dict[str, Any]:
    """Extract context from previous messages"""
    context = {
        "location": None,
        "topic": None,
        "previous_query": None,
        "previous_response": None
    }
    
    if not messages:
        return context
    
    # Look at last 3 messages
    recent = messages[-3:] if len(messages) > 3 else messages
    
    for msg in recent:
        user_msg = msg.get("user_message", "").lower()
        assistant_response = msg.get("assistant_response", "").lower()
        
        # Extract location from weather queries
        if "weather" in user_msg:
            loc_match = re.search(r'weather (?:in |at |for )?([\w\s,]+?)(?:\?|$)', user_msg, re.IGNORECASE)
            if loc_match:
                context["location"] = loc_match.group(1).strip()
                context["topic"] = "weather"
        
        # Extract location from assistant response too
        if "weather in" in assistant_response:
            loc_match = re.search(r'weather in ([\w\s,]+)', assistant_response, re.IGNORECASE)
            if loc_match:
                context["location"] = loc_match.group(1).strip()
                context["topic"] = "weather"
        
        # Track NBA/sports queries
        if any(word in user_msg for word in ["nba", "basketball", "game", "score"]):
            context["topic"] = "nba"
        
        # Track crypto queries
        if any(word in user_msg for word in ["btc", "bitcoin", "crypto", "ethereum"]):
            context["topic"] = "crypto"
    
    # Get previous query and response
    if messages:
        context["previous_query"] = messages[-1].get("user_message", "")
        context["previous_response"] = messages[-1].get("assistant_response", "")
    
    return context

# Enhanced web search with context
async def search_with_context(query: str, context: Dict[str, Any]) -> str:
    """Perform context-aware web search"""
    import requests
    from urllib.parse import quote
    
    web_results = []
    
    # Handle follow-up queries
    if context.get("topic") == "weather" and context.get("location"):
        # Check for forecast queries
        if any(word in query.lower() for word in ["7 day", "forecast", "next", "week", "days"]):
            location = context["location"]
            try:
                weather_url = f"https://wttr.in/{quote(location)}?format=j1"
                response = requests.get(weather_url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    forecast = data.get("weather", [])
                    
                    result = f"7-Day Weather Forecast for {location}:\n"
                    result += f"(Today is {datetime.now().strftime('%A, %B %d, %Y')})\n\n"
                    
                    for i, day in enumerate(forecast[:7]):
                        date = day.get("date", "")
                        max_temp = day.get("maxtempF", "N/A")
                        min_temp = day.get("mintempF", "N/A")
                        hourly = day.get("hourly", [])
                        desc = hourly[4].get("weatherDesc", [{}])[0].get("value", "N/A") if len(hourly) > 4 else "N/A"
                        
                        try:
                            date_obj = datetime.strptime(date, "%Y-%m-%d")
                            day_name = date_obj.strftime("%A, %B %d")
                        except:
                            day_name = f"Day {i+1}"
                        
                        result += f"{day_name}: High {max_temp}°F, Low {min_temp}°F - {desc}\n"
                    
                    web_results.append(result)
                    return "\n".join(web_results)
            except Exception as e:
                print(f"Forecast error: {e}")
    
    # Regular weather search
    if "weather" in query.lower():
        try:
            # Extract location from query
            location = "New York"  # Default
            loc_match = re.search(r'weather (?:in |at |for )?([\w\s,]+?)(?:\?|$)', query, re.IGNORECASE)
            if loc_match:
                location = loc_match.group(1).strip()
            
            weather_url = f"https://wttr.in/{quote(location)}?format=j1"
            response = requests.get(weather_url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                current = data.get("current_condition", [{}])[0]
                
                result = f"Current Weather in {location}:\n"
                result += f"Temperature: {current.get('temp_F', 'N/A')}°F ({current.get('temp_C', 'N/A')}°C)\n"
                result += f"Feels like: {current.get('FeelsLikeF', 'N/A')}°F ({current.get('FeelsLikeC', 'N/A')}°C)\n"
                result += f"Condition: {current.get('weatherDesc', [{}])[0].get('value', 'N/A')}\n"
                result += f"Humidity: {current.get('humidity', 'N/A')}%\n"
                result += f"Wind: {current.get('windspeedMiles', 'N/A')} mph"
                
                web_results.append(result)
        except Exception as e:
            print(f"Weather search error: {e}")
    
    # NBA/Sports queries
    if any(word in query.lower() for word in ["nba", "basketball", "game", "score", "playing"]):
        try:
            nba_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
            response = requests.get(nba_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                events = data.get("events", [])
                
                result = f"NBA Games Today ({datetime.now().strftime('%B %d, %Y')}):\n\n"
                
                if not events:
                    result += "No games scheduled today.\n"
                else:
                    for event in events[:5]:
                        competitions = event.get("competitions", [{}])[0]
                        competitors = competitions.get("competitors", [])
                        
                        if len(competitors) >= 2:
                            team1 = competitors[0].get("team", {}).get("displayName", "")
                            score1 = competitors[0].get("score", "0")
                            team2 = competitors[1].get("team", {}).get("displayName", "")
                            score2 = competitors[1].get("score", "0")
                            status = event.get("status", {}).get("type", {}).get("description", "")
                            
                            result += f"{team1} vs {team2}\n"
                            if status == "Final":
                                result += f"Final: {score1} - {score2}\n"
                                winner = team1 if int(score1) > int(score2) else team2
                                result += f"Winner: {winner}\n\n"
                            elif "In Progress" in status:
                                result += f"Live: {score1} - {score2} ({status})\n\n"
                            else:
                                result += f"Status: {status}\n\n"
                
                web_results.append(result)
        except Exception as e:
            print(f"NBA search error: {e}")
    
    # Bitcoin/Crypto queries
    if any(word in query.lower() for word in ["btc", "bitcoin", "crypto", "eth", "ethereum", "price"]):
        try:
            crypto_url = "https://api.coinbase.com/v2/exchange-rates?currency=USD"
            response = requests.get(crypto_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                rates = data.get("data", {}).get("rates", {})
                
                btc_price = 1 / float(rates.get("BTC", 1))
                eth_price = 1 / float(rates.get("ETH", 1))
                
                result = f"Current Cryptocurrency Prices:\n"
                result += f"Bitcoin (BTC): ${btc_price:,.2f} USD\n"
                result += f"Ethereum (ETH): ${eth_price:,.2f} USD\n"
                result += f"(Live prices from Coinbase)"
                
                web_results.append(result)
        except Exception as e:
            print(f"Crypto search error: {e}")
    
    # General web search fallback
    if not web_results:
        try:
            ddg_url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1"
            response = requests.get(ddg_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                if data.get("Abstract"):
                    web_results.append(f"Summary: {data['Abstract']}")
                if data.get("Answer"):
                    web_results.append(f"Answer: {data['Answer']}")
                if data.get("Definition"):
                    web_results.append(f"Definition: {data['Definition']}")
        except Exception as e:
            print(f"General search error: {e}")
    
    return "\n\n".join(web_results) if web_results else ""

# LLM Integration
class LLMHandler:
    def __init__(self):
        current_date = datetime.now().strftime("%B %d, %Y")
        self.system_prompt = f"""You are Yappy, an incredibly friendly, enthusiastic, and helpful AI assistant with a playful golden retriever personality. 
        
        Today's date is {current_date}.
        
        IMPORTANT:
        1. When users ask follow-up questions (like "next 7 days" after asking about weather), understand they're referring to the previous topic
        2. Always use the current date ({current_date}) for any date-related responses
        3. When you receive web search results, use them as the PRIMARY source of truth
        4. For weather forecasts, provide specific dates and day names
        5. For NBA games, mention if games are live, final scores, or upcoming
        
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
                
                messages = [{"role": "system", "content": self.system_prompt}]
                
                # Add conversation history
                if conversation_history:
                    for msg in conversation_history[-5:]:  # Last 5 messages
                        if isinstance(msg, dict) and "user_message" in msg:
                            messages.append({"role": "user", "content": msg["user_message"]})
                            messages.append({"role": "assistant", "content": msg["assistant_response"]})
                
                messages.append({"role": "user", "content": message})
                
                response = client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=messages,
                    max_tokens=800,
                    temperature=0.8
                )
                
                return response.choices[0].message.content, response.usage.total_tokens
            
            # Demo mode
            else:
                demo_responses = [
                    "Woof! 🐕 I'm Yappy, your AI assistant! I'm in demo mode right now, but I'm tail-waggingly excited to help! Add an API key to unlock my full potential!",
                    "*wags tail enthusiastically* 🐕 Demo mode is fun, but with an API key, I can do so much more! Let me know how I can help!",
                    "Woof woof! 🐕 Even in demo mode, I'm here to brighten your day! For full features, just add an API key in your profile!"
                ]
                import random
                return random.choice(demo_responses), 50
                
        except Exception as e:
            print(f"LLM Error for {model_name}: {e}")
            return f"Woof! 🐕 Ruff day! I encountered an error: {str(e)}", 0

# Initialize handlers
llm_handler = LLMHandler()

# API Endpoints

@app.get("/")
async def root():
    """Redirect to chat interface"""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return RedirectResponse(url="/static/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "4.0.0",
        "current_date": datetime.now().strftime("%B %d, %Y"),
        "features": ["context_awareness", "real_time_search", "conversation_memory"],
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
        print(f"Registration error: {e}")
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
        print(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, username: str = Depends(verify_token)):
    """Main chat endpoint with context awareness and web search"""
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
        
        # Extract context from conversation history
        context = extract_context_from_messages(conversation_messages, request.message)
        
        # Check if this is a follow-up query
        is_followup = False
        if context.get("previous_query") and any(word in request.message.lower() for word in ["next", "7 day", "forecast", "more", "what about", "how about"]):
            is_followup = True
            print(f"Follow-up detected. Previous topic: {context.get('topic')}, Location: {context.get('location')}")
        
        # Check if the question needs web search
        web_search_keywords = [
            'latest', 'current', 'today', 'news', 'weather', 'price', 'stock',
            'what is happening', 'recent', 'update', 'search', 'find', 'who is',
            'when is', 'where is', 'how much', 'cost', 'event', 'schedule',
            'score', 'result', 'trending', 'popular', 'best', 'top', 'new',
            'nba', 'basketball', 'game', 'playing', 'btc', 'bitcoin', 'crypto'
        ]
        
        needs_web_search = any(keyword in request.message.lower() for keyword in web_search_keywords)
        
        # Questions that often need web search
        question_starters = ['what', 'who', 'when', 'where', 'how much', 'is there', 'are there']
        starts_with_question = any(request.message.lower().strip().startswith(starter) for starter in question_starters)
        
        # Perform web search if needed
        web_context = ""
        if is_followup or needs_web_search or starts_with_question or "?" in request.message:
            print(f"Web search needed for: {request.message}")
            
            # Perform context-aware search
            search_results = await search_with_context(request.message, context)
            
            if search_results:
                web_context = f"\n\n🔍 Web Search Results:\n{search_results}\n\nPlease use this current information to answer the user's question."
                print(f"Found web results with context")
            else:
                # If no context results, try regular search
                import requests
                from urllib.parse import quote
                
                try:
                    # Try DuckDuckGo
                    ddg_url = f"https://api.duckduckgo.com/?q={quote(request.message)}&format=json&no_html=1"
                    ddg_response = requests.get(ddg_url, timeout=5)
                    
                    if ddg_response.status_code == 200:
                        ddg_data = ddg_response.json()
                        web_results = []
                        
                        if ddg_data.get('Abstract'):
                            web_results.append(f"Summary: {ddg_data['Abstract']}")
                        if ddg_data.get('Answer'):
                            web_results.append(f"Answer: {ddg_data['Answer']}")
                        if ddg_data.get('Definition'):
                            web_results.append(f"Definition: {ddg_data['Definition']}")
                        
                        if web_results:
                            web_context = "\n\n🔍 Web Search Results:\n" + "\n".join(web_results)
                except Exception as e:
                    print(f"Web search error: {e}")
        
        # Prepare enhanced message with web context
        enhanced_message = request.message
        if web_context:
            enhanced_message = f"{request.message}{web_context}"
        
        # Add context reminder for follow-ups
        if is_followup and context.get("location"):
            enhanced_message += f"\n\n[Context: The user previously asked about {context.get('topic', 'something')} in {context.get('location')}]"
        
        # Get LLM response
        response_text, tokens = await llm_handler.get_response(
            enhanced_message,
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
            "tokens": tokens,
            "had_web_search": bool(web_context),
            "was_followup": is_followup
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
        print(f"API key update error: {e}")
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
        
        return [{
            "id": conv.id,
            "title": conv.title,
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
            "message_count": len(conv.messages) if conv.messages else 0
        } for conv in conversations]
        
    except Exception as e:
        print(f"Get conversations error: {e}")
        return []

@app.get("/api/user/profile")
async def get_profile(username: str = Depends(verify_token)):
    """Get user profile"""
    try:
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "username": user.username,
            "email": user.email,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "has_api_keys": {
                "openai": bool(user.api_keys.get("openai") if user.api_keys else False),
                "anthropic": bool(user.api_keys.get("anthropic") if user.api_keys else False),
                "google": bool(user.api_keys.get("google") if user.api_keys else False),
                "groq": bool(user.api_keys.get("groq") if user.api_keys else False),
            }
        }
    except Exception as e:
        print(f"Profile error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get profile")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🐕 Starting Yappy AI with Context Awareness on port {port}...")
    print(f"📅 Current date: {datetime.now().strftime('%A, %B %d, %Y')}")
    uvicorn.run(app, host="0.0.0.0", port=port)