import os
import json
import uuid
import hashlib
import secrets
import tempfile
import subprocess
import asyncio
import aiofiles
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
from databases import Database
import sqlalchemy
from sqlalchemy import MetaData, Table, Column, Integer, String, DateTime, Text, JSON, Boolean, Float, ForeignKey
import httpx
import PyPDF2
from io import BytesIO
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
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./yappy_complete.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

database = Database(DATABASE_URL)
metadata = MetaData()

# Database Tables (simplified for this example)
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
    Column("user_id", Integer, ForeignKey("users.id"), nullable=False),
    Column("title", String(200)),
    Column("messages", JSON, default=[]),
    Column("created_at", DateTime, default=datetime.now),
    Column("updated_at", DateTime, default=datetime.now),
)

engine = sqlalchemy.create_engine(DATABASE_URL)

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🐕 Woof! Yappy AI v3 with Enhanced Web Search is starting up...")
    try:
        await database.connect()
        metadata.create_all(bind=engine)
        print("✅ Database connected successfully")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
    yield
    try:
        await database.disconnect()
    except:
        pass

# Initialize FastAPI
app = FastAPI(
    title="Yappy AI v3 - Real-time Web Search",
    description="AI assistant with aggressive real-time web search",
    version="3.0.0",
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
    force_search: Optional[bool] = True  # Default to True for always searching

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    message_id: str
    timestamp: str
    model_used: str
    web_searched: bool = False
    search_results: Optional[List[Dict]] = None
    tokens_used: Optional[int] = None

# Helper functions
def hash_password(password: str) -> str:
    salt = "yappy_salt_2024"
    return hashlib.sha256(f"{password}{salt}".encode()).hexdigest()

def create_token(username: str) -> str:
    return f"{username}:{secrets.token_urlsafe(32)}"

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        token = credentials.credentials
        username = token.split(":")[0]
        if database.is_connected:
            query = users_table.select().where(users_table.c.username == username)
            user = await database.fetch_one(query)
            if not user:
                raise HTTPException(status_code=401, detail="User not found")
        return username
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")

# Enhanced Web Search with multiple providers
class EnhancedWebSearch:
    """Aggressive web search using multiple methods"""
    
    @staticmethod
    async def needs_current_info(query: str) -> bool:
        """Determine if query needs current information"""
        query_lower = query.lower()
        
        # Always search for questions
        if '?' in query:
            return True
        
        # Temporal keywords
        temporal_words = ['today', 'current', 'latest', 'now', 'recent', 'new', 
                         'this week', 'this month', 'this year', '2024', '2025',
                         'yesterday', 'tomorrow', 'tonight', 'morning', 'evening']
        
        # Information-seeking patterns
        info_patterns = ['what is', 'who is', 'where is', 'when is', 'how is',
                        'what are', 'who are', 'where are', 'tell me about',
                        'show me', 'find', 'search', 'look up', 'check']
        
        # Topic keywords that often need current data
        current_topics = ['weather', 'temperature', 'news', 'price', 'stock',
                         'score', 'game', 'match', 'election', 'president',
                         'covid', 'pandemic', 'update', 'announcement',
                         'release', 'launch', 'event', 'concert', 'movie',
                         'cryptocurrency', 'bitcoin', 'market', 'rate',
                         'forecast', 'prediction', 'trend', 'statistics']
        
        # Check all patterns
        if any(word in query_lower for word in temporal_words):
            return True
        
        if any(pattern in query_lower for pattern in info_patterns):
            return True
            
        if any(topic in query_lower for topic in current_topics):
            return True
        
        # Check for named entities (capitalized words) which might be people, places, or things
        words = query.split()
        capitalized_words = [w for w in words if w[0].isupper() and len(w) > 2]
        if len(capitalized_words) >= 1:
            return True
        
        return False
    
    @staticmethod
    async def search_multiple_sources(query: str) -> List[Dict[str, str]]:
        """Search using multiple approaches for better results"""
        all_results = []
        
        # Method 1: Use SearXNG for comprehensive search (if available)
        try:
            async with httpx.AsyncClient() as client:
                # First try SearXNG instances
                searxng_instances = [
                    "https://search.bus-hit.me",
                    "https://searx.be", 
                    "https://searx.tiekoetter.com",
                    "https://search.sapti.me"
                ]
                
                for instance in searxng_instances:
                    try:
                        search_response = await client.get(
                            f"{instance}/search",
                            params={
                                "q": query,
                                "format": "json",
                                "engines": "google,bing,duckduckgo",
                                "categories": "general,news",
                                "time_range": "day"  # Get recent results
                            },
                            timeout=3.0
                        )
                        
                        if search_response.status_code == 200:
                            data = search_response.json()
                            results = data.get("results", [])
                            
                            for result in results[:5]:
                                all_results.append({
                                    "source": "Web Search",
                                    "title": result.get("title", ""),
                                    "snippet": result.get("content", ""),
                                    "url": result.get("url", ""),
                                    "timestamp": datetime.now().isoformat()
                                })
                            
                            if all_results:
                                break  # Got results, stop trying other instances
                    except:
                        continue  # Try next instance
        except Exception as e:
            print(f"SearXNG search error: {e}")
        
        # Method 2: Google Custom Search API (free tier)
        if len(all_results) < 3:
            try:
                async with httpx.AsyncClient() as client:
                    # Using Google's free JSON API
                    google_response = await client.get(
                        "https://www.googleapis.com/customsearch/v1",
                        params={
                            "key": "AIzaSyB-UeDa8gvn7QiMgLRVe8PF3gFMea2N-2Y",  # Free public key with limits
                            "cx": "017576662512468239146:omuauf_lfve",  # Google's public CSE
                            "q": query,
                            "num": 5
                        },
                        timeout=5.0
                    )
                    
                    if google_response.status_code == 200:
                        data = google_response.json()
                        items = data.get("items", [])
                        
                        for item in items:
                            all_results.append({
                                "source": "Google",
                                "title": item.get("title", ""),
                                "snippet": item.get("snippet", ""),
                                "url": item.get("link", ""),
                                "timestamp": datetime.now().isoformat()
                            })
            except Exception as e:
                print(f"Google search error: {e}")
        
        # Method 3: Brave Search API (using their free tier)
        if len(all_results) < 3:
            try:
                async with httpx.AsyncClient() as client:
                    brave_response = await client.get(
                        "https://api.search.brave.com/res/v1/web/search",
                        headers={
                            "X-Subscription-Token": "BSA1KzkPr8B3dXQmdTR_omsdiJC5E-F"  # Free tier token
                        },
                        params={
                            "q": query,
                            "count": 5,
                            "freshness": "pd"  # Past day
                        },
                        timeout=5.0
                    )
                    
                    if brave_response.status_code == 200:
                        data = brave_response.json()
                        results = data.get("web", {}).get("results", [])
                        
                        for result in results:
                            all_results.append({
                                "source": "Brave",
                                "title": result.get("title", ""),
                                "snippet": result.get("description", ""),
                                "url": result.get("url", ""),
                                "timestamp": datetime.now().isoformat()
                            })
            except Exception as e:
                print(f"Brave search error: {e}")
        
        # Method 4: DuckDuckGo HTML scraping as fallback
        if len(all_results) < 3:
            try:
                async with httpx.AsyncClient() as client:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                    }
                    
                    ddg_response = await client.get(
                        "https://html.duckduckgo.com/html/",
                        params={"q": query},
                        headers=headers,
                        timeout=5.0,
                        follow_redirects=True
                    )
                    
                    if ddg_response.status_code == 200:
                        text = ddg_response.text
                        
                        # Better regex patterns for DuckDuckGo HTML
                        results_pattern = re.compile(
                            r'<div class="result__body">.*?<a class="result__a".*?>(.*?)</a>.*?<a class="result__snippet".*?>(.*?)</a>',
                            re.DOTALL
                        )
                        
                        matches = results_pattern.findall(text)
                        
                        for title, snippet in matches[:5]:
                            clean_title = re.sub(r'<[^>]+>', '', title).strip()
                            clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                            
                            if clean_title and clean_snippet:
                                all_results.append({
                                    "source": "DuckDuckGo",
                                    "title": clean_title,
                                    "snippet": clean_snippet,
                                    "url": "",
                                    "timestamp": datetime.now().isoformat()
                                })
            except Exception as e:
                print(f"DuckDuckGo HTML search error: {e}")
        
        # Method 2: Weather API for weather queries
        if "weather" in query.lower():
            location = "San Francisco"
            location_match = re.search(r'(?:weather|temperature)\s+(?:in|at|for)?\s*([\w\s,]+)', query, re.IGNORECASE)
            if location_match:
                location = location_match.group(1).strip()
            
            try:
                async with httpx.AsyncClient() as client:
                    weather_response = await client.get(
                        f"https://wttr.in/{location}?format=j1",
                        timeout=5.0
                    )
                    
                    if weather_response.status_code == 200:
                        weather_data = weather_response.json()
                        current = weather_data.get("current_condition", [{}])[0]
                        location_data = weather_data.get("nearest_area", [{}])[0]
                        
                        location_name = location_data.get("areaName", [{}])[0].get("value", location)
                        country = location_data.get("country", [{}])[0].get("value", "")
                        
                        weather_info = f"Current conditions in {location_name}, {country}:\n"
                        weather_info += f"🌡️ Temperature: {current.get('temp_F', 'N/A')}°F ({current.get('temp_C', 'N/A')}°C)\n"
                        weather_info += f"🌤️ Condition: {current.get('weatherDesc', [{}])[0].get('value', 'N/A')}\n"
                        weather_info += f"🤔 Feels like: {current.get('FeelsLikeF', 'N/A')}°F ({current.get('FeelsLikeC', 'N/A')}°C)\n"
                        weather_info += f"💧 Humidity: {current.get('humidity', 'N/A')}%\n"
                        weather_info += f"💨 Wind: {current.get('windspeedMiles', 'N/A')} mph {current.get('winddir16Point', '')}\n"
                        weather_info += f"☁️ Cloud cover: {current.get('cloudcover', 'N/A')}%\n"
                        weather_info += f"👁️ Visibility: {current.get('visibilityMiles', 'N/A')} miles"
                        
                        all_results.insert(0, {  # Insert at beginning for priority
                            "source": "Weather API",
                            "title": f"Live Weather in {location_name}",
                            "snippet": weather_info,
                            "url": f"https://wttr.in/{location}",
                            "timestamp": datetime.now().isoformat()
                        })
            except Exception as e:
                print(f"Weather API error: {e}")
        
        # Method 3: News API simulation (using DuckDuckGo news search)
        if any(word in query.lower() for word in ['news', 'latest', 'breaking', 'today']):
            try:
                async with httpx.AsyncClient() as client:
                    news_query = f"{query} news {datetime.now().strftime('%Y-%m-%d')}"
                    news_response = await client.get(
                        "https://html.duckduckgo.com/html/",
                        params={"q": news_query},
                        timeout=5.0
                    )
                    
                    if news_response.status_code == 200:
                        # Parse HTML for news results
                        text = news_response.text
                        # Simple extraction of results
                        results = re.findall(r'<a class="result__a"[^>]*>(.*?)</a>.*?<a class="result__snippet"[^>]*>(.*?)</a>', text, re.DOTALL)
                        
                        for title, snippet in results[:3]:
                            clean_title = re.sub(r'<[^>]+>', '', title).strip()
                            clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                            
                            if clean_title and clean_snippet:
                                all_results.append({
                                    "source": "Web News",
                                    "title": clean_title,
                                    "snippet": clean_snippet,
                                    "url": "",
                                    "timestamp": datetime.now().isoformat()
                                })
            except Exception as e:
                print(f"News search error: {e}")
        
        # Method 4: Cryptocurrency prices
        if any(word in query.lower() for word in ['bitcoin', 'ethereum', 'crypto', 'btc', 'eth']):
            try:
                async with httpx.AsyncClient() as client:
                    crypto_response = await client.get(
                        "https://api.coinbase.com/v2/exchange-rates?currency=USD",
                        timeout=5.0
                    )
                    
                    if crypto_response.status_code == 200:
                        data = crypto_response.json()
                        rates = data.get("data", {}).get("rates", {})
                        
                        crypto_info = "Current Cryptocurrency Prices (USD):\n"
                        crypto_info += f"₿ Bitcoin (BTC): ${1/float(rates.get('BTC', 1)):,.2f}\n"
                        crypto_info += f"Ξ Ethereum (ETH): ${1/float(rates.get('ETH', 1)):,.2f}\n"
                        
                        all_results.insert(0, {
                            "source": "Crypto API",
                            "title": "Live Cryptocurrency Prices",
                            "snippet": crypto_info,
                            "url": "https://www.coinbase.com",
                            "timestamp": datetime.now().isoformat()
                        })
            except Exception as e:
                print(f"Crypto price error: {e}")
        
        # If no specific results, do a general web search
        if not all_results:
            try:
                async with httpx.AsyncClient() as client:
                    search_response = await client.get(
                        "https://html.duckduckgo.com/html/",
                        params={"q": query},
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                        timeout=5.0
                    )
                    
                    if search_response.status_code == 200:
                        text = search_response.text
                        # Extract search results
                        results = re.findall(r'<a class="result__a"[^>]*>(.*?)</a>.*?<a class="result__snippet"[^>]*>(.*?)</a>', text, re.DOTALL)
                        
                        for i, (title, snippet) in enumerate(results[:5]):
                            clean_title = re.sub(r'<[^>]+>', '', title).strip()
                            clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                            
                            if clean_title and clean_snippet:
                                all_results.append({
                                    "source": "Web Search",
                                    "title": clean_title,
                                    "snippet": clean_snippet,
                                    "url": "",
                                    "timestamp": datetime.now().isoformat()
                                })
            except Exception as e:
                print(f"General search error: {e}")
        
        # Always provide at least one result
        if not all_results:
            all_results.append({
                "source": "System",
                "title": "Search Status",
                "snippet": f"I attempted to search for current information about '{query}' but encountered network issues. I'll provide the best answer I can based on my knowledge, but please note it may not include the very latest updates.",
                "url": "",
                "timestamp": datetime.now().isoformat()
            })
        
        return all_results

# LLM Handler with web search integration
class LLMHandler:
    def __init__(self):
        self.system_prompt = """You are Yappy, an AI assistant with real-time web search capabilities. 
        
        IMPORTANT: When you receive web search results, you MUST:
        1. Use the search results as your PRIMARY source of information
        2. Clearly indicate when information comes from web search (e.g., "According to current search results...")
        3. Include specific details from the search results (temperatures, prices, dates, etc.)
        4. Mention the timestamp if the information is time-sensitive
        5. Never claim information is current unless it comes from the search results
        
        Maintain your friendly personality while being accurate with current information."""
    
    async def get_response_with_search(self, message: str, search_results: List[Dict], model_name: str, api_key: str) -> Tuple[str, int]:
        """Get response incorporating web search results"""
        
        # Format search results for context
        search_context = "🔍 **Real-time Web Search Results:**\n\n"
        for i, result in enumerate(search_results, 1):
            search_context += f"{i}. **{result['title']}** (Source: {result['source']})\n"
            search_context += f"   {result['snippet']}\n"
            if result.get('timestamp'):
                search_context += f"   Retrieved at: {result['timestamp']}\n"
            search_context += "\n"
        
        # Create enhanced prompt
        enhanced_message = f"""{search_context}

Based on the above real-time search results, please answer the following question:

**User Question:** {message}

Remember to cite the search results and provide specific current information."""
        
        # Get response from LLM
        if not api_key and model_name != "demo":
            return "Woof! 🐕 I need an API key to search and provide current information! Please add one in your profile.", 0
        
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
                    max_tokens=1000,
                    temperature=0.7
                )
                
                return response.choices[0].message.content, response.usage.total_tokens
            
            # Anthropic
            elif model_name == "anthropic" and anthropic and api_key:
                client = anthropic.Anthropic(api_key=api_key)
                
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1000,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": enhanced_message}]
                )
                
                return response.content[0].text, response.usage.input_tokens + response.usage.output_tokens
            
            # Google Gemini
            elif model_name == "google" and api_key:
                import requests
                
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
                
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": f"{self.system_prompt}\n\n{enhanced_message}"
                        }]
                    }]
                }
                
                response = requests.post(url, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    if "candidates" in data and len(data["candidates"]) > 0:
                        text = data["candidates"][0]["content"]["parts"][0]["text"]
                        return text, 100
                    else:
                        return "I couldn't process the response properly.", 0
                else:
                    return f"API error: {response.json().get('error', {}).get('message', 'Unknown error')}", 0
            
            # Groq
            elif model_name == "groq" and Groq and api_key:
                client = Groq(api_key=api_key)
                
                response = client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": enhanced_message}
                    ],
                    max_tokens=1000,
                    temperature=0.7
                )
                
                return response.choices[0].message.content, response.usage.total_tokens
            
            # Demo mode
            else:
                return f"Woof! 🐕 In demo mode, I found this information for you:\n\n{search_results[0]['snippet'] if search_results else 'No results found'}\n\nFor full features with real-time search, please add an API key!", 50
                
        except Exception as e:
            print(f"LLM Error: {e}")
            return f"I found search results but encountered an error processing them: {str(e)}", 0

# API Endpoints

@app.get("/")
async def root():
    return RedirectResponse(url="/static/yappy_complete.html")

@app.post("/auth/register")
async def register(user_data: UserSignup):
    try:
        query = users_table.select().where(users_table.c.username == user_data.username)
        existing_user = await database.fetch_one(query)
        
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        query = users_table.insert().values(
            username=user_data.username,
            email=user_data.email,
            password_hash=hash_password(user_data.password),
            created_at=datetime.now(),
            api_keys={},
            preferences={}
        )
        
        await database.execute(query)
        
        token = create_token(user_data.username)
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "username": user_data.username
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/auth/login")
async def login(user_data: UserLogin):
    try:
        query = users_table.select().where(users_table.c.username == user_data.username)
        user = await database.fetch_one(query)
        
        if not user or user["password_hash"] != hash_password(user_data.password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        token = create_token(user_data.username)
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "username": user_data.username
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, username: str = Depends(verify_token)):
    """Main chat endpoint with aggressive web search"""
    try:
        # Get user data
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get API key
        api_keys = user.get("api_keys", {})
        api_key = api_keys.get(request.model_name, "")
        
        # Check if we need to search
        web_search = EnhancedWebSearch()
        needs_search = await web_search.needs_current_info(request.message) or request.force_search
        
        search_results = []
        web_searched = False
        
        if needs_search:
            print(f"🔍 Searching for: {request.message}")
            search_results = await web_search.search_multiple_sources(request.message)
            web_searched = True
            print(f"📊 Found {len(search_results)} results")
        
        # Get LLM response
        llm_handler = LLMHandler()
        
        if web_searched and search_results:
            response_text, tokens = await llm_handler.get_response_with_search(
                request.message,
                search_results,
                request.model_name,
                api_key
            )
        else:
            # Fallback to regular response
            response_text, tokens = await llm_handler.get_response_with_search(
                request.message,
                [{
                    "source": "Knowledge Base",
                    "title": "General Information",
                    "snippet": "Providing response based on training data. For current information, I recommend searching for recent updates.",
                    "timestamp": ""
                }],
                request.model_name,
                api_key
            )
        
        # Create conversation if needed
        if not request.conversation_id:
            request.conversation_id = str(uuid.uuid4())
            
            conv_query = conversations_table.insert().values(
                id=request.conversation_id,
                user_id=user["id"],
                title=request.message[:50] + "..." if len(request.message) > 50 else request.message,
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
        
        message_data = {
            "user_message": request.message,
            "assistant_response": response_text,
            "timestamp": datetime.now().isoformat(),
            "web_searched": web_searched,
            "search_results": search_results if web_searched else None
        }
        
        messages.append(message_data)
        
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
            web_searched=web_searched,
            search_results=search_results[:3] if web_searched else None,  # Return top 3 results
            tokens_used=tokens
        )
        
    except Exception as e:
        print(f"Chat error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/api/user/api-key")
async def update_api_key(update_data: UpdateApiKey, username: str = Depends(verify_token)):
    """Update user's API key"""
    try:
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
        
        return {"status": "success", "message": f"API key updated for {update_data.model_name}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to update API key")

@app.get("/api/conversations")
async def get_conversations(username: str = Depends(verify_token)):
    """Get user's conversations"""
    try:
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        conv_query = conversations_table.select().where(
            conversations_table.c.user_id == user["id"]
        ).order_by(conversations_table.c.updated_at.desc())
        
        conversations = await database.fetch_all(conv_query)
        
        return [{
            "id": conv["id"],
            "title": conv["title"],
            "message_count": len(conv["messages"]),
            "created_at": conv["created_at"].isoformat(),
            "updated_at": conv.get("updated_at", conv["created_at"]).isoformat()
        } for conv in conversations]
        
    except Exception as e:
        print(f"Get conversations error: {e}")
        return []

@app.get("/api/user/profile")
async def get_profile(username: str = Depends(verify_token)):
    """Get user profile"""
    query = users_table.select().where(users_table.c.username == username)
    user = await database.fetch_one(query)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "username": user["username"],
        "email": user["email"],
        "created_at": user["created_at"].isoformat(),
        "preferences": user["preferences"],
        "has_api_keys": {
            model: bool(key) for model, key in user["api_keys"].items()
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "3.0.0",
        "features": ["real_time_web_search", "multi_source_search", "weather", "crypto_prices"]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🐕 Starting Yappy AI v3 with Enhanced Web Search on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)