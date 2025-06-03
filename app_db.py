"""
Yappy AI Assistant - Fixed to match local app's agent routing
Web queries go to browser agent, not casual agent
"""
import os
import json
import uuid
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv
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
from bs4 import BeautifulSoup

# Load environment variables
from pathlib import Path
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug: Check if API keys are loaded
logger.info(f"OPENAI_API_KEY loaded: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
logger.info(f"CLAUDE_API_KEY loaded: {'Yes' if os.getenv('CLAUDE_API_KEY') else 'No'}")

# Import optional libraries
try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available - image processing features will be limited")

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

# Improved Search Implementation with weather support
class ImprovedSearch:
    def __init__(self):
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    
    def execute(self, query: str) -> str:
        """Execute search with smart routing"""
        query_lower = query.lower()
        
        # Check if it's a weather query
        if 'weather' in query_lower or 'temperature' in query_lower or 'forecast' in query_lower:
            return self._get_weather_data(query)
        
        # For other queries, try web search
        return self._web_search(query)
    
    def _get_weather_data(self, query: str) -> str:
        """Get weather data directly from wttr.in"""
        # Extract location from query
        location = self._extract_location(query)
        
        try:
            from urllib.parse import quote
            # Use wttr.in API which doesn't require authentication
            weather_url = f"https://wttr.in/{quote(location)}?format=j1"
            response = requests.get(weather_url, timeout=10, headers={'User-Agent': self.user_agent})
            
            if response.status_code == 200:
                data = response.json()
                current = data.get('current_condition', [{}])[0]
                location_data = data.get('nearest_area', [{}])[0]
                
                # Format location
                city = location_data.get('areaName', [{}])[0].get('value', location)
                region = location_data.get('region', [{}])[0].get('value', '')
                country = location_data.get('country', [{}])[0].get('value', '')
                
                # Format weather data
                result = f"Title:Current Weather in {city}, {region}\n"
                result += f"Snippet:Temperature: {current.get('temp_F', 'N/A')}°F ({current.get('temp_C', 'N/A')}°C)\n"
                result += f"Feels like: {current.get('FeelsLikeF', 'N/A')}°F ({current.get('FeelsLikeC', 'N/A')}°C)\n"
                result += f"Condition: {current.get('weatherDesc', [{}])[0].get('value', 'N/A')}\n"
                result += f"Humidity: {current.get('humidity', 'N/A')}%\n"
                result += f"Wind: {current.get('windspeedMiles', 'N/A')} mph {current.get('winddir16Point', '')}\n"
                result += f"UV Index: {current.get('uvIndex', 'N/A')}\n"
                result += f"Visibility: {current.get('visibility', 'N/A')} miles\n"
                result += f"Pressure: {current.get('pressure', 'N/A')} mb"
                result += f"\nLink:https://wttr.in/{quote(location)}"
                
                # Add forecast
                forecast_data = []
                weather = data.get('weather', [])
                for i, day in enumerate(weather[:3]):  # Next 3 days
                    date = day.get('date', '')
                    max_temp = day.get('maxtempF', 'N/A')
                    min_temp = day.get('mintempF', 'N/A')
                    desc = day.get('hourly', [{}])[4].get('weatherDesc', [{}])[0].get('value', 'N/A') if day.get('hourly') else 'N/A'
                    forecast_data.append(f"Day {i+1} ({date}): High {max_temp}°F, Low {min_temp}°F - {desc}")
                
                if forecast_data:
                    result += f"\n\nTitle:3-Day Forecast\nSnippet:" + "\n".join(forecast_data)
                    result += "\nLink:"
                
                return result
                
        except Exception as e:
            logger.error(f"Weather API error: {e}")
        
        # Fallback to web search for weather
        return self._web_search(query)
    
    def _extract_location(self, query: str) -> str:
        """Extract location from weather query"""
        # Original query for reference
        original = query
        
        # Remove common words but preserve "in" followed by location
        query = query.lower()
        
        # Try to find location after "in"
        if " in " in query:
            parts = query.split(" in ", 1)
            if len(parts) > 1:
                location = parts[1].strip()
                # Remove trailing punctuation
                location = location.rstrip("?.,!").strip()
                if location:
                    return location
        
        # Fallback: remove weather-related words
        query = query.replace("what's the", "").replace("what is the", "")
        query = query.replace("how's the", "").replace("how is the", "")
        query = query.replace("weather", "").replace("temperature", "")
        query = query.replace("forecast", "").replace("today", "")
        query = query.replace("?", "").strip()
        
        # Clean up extra spaces
        location = " ".join(query.split())
        
        # Default to a location if empty
        if not location:
            location = "New York"
        
        return location
    
    def _web_search(self, query: str) -> str:
        """Perform general web search with special handling for sports/NBA"""
        query_lower = query.lower()
        
        # Special handling for NBA/sports queries
        nba_keywords = ['nba', 'basketball', 'lakers', 'warriors', 'celtics', 'knicks', 'pacers', 
                       'who is playing', 'who\'s playing', 'game today', 'games today', 'score', 
                       'playoff', 'finals']
        if any(term in query_lower for term in nba_keywords):
            nba_results = self._get_nba_scores()
            if nba_results:
                # Return NBA scores immediately without additional web search
                # to avoid confusion with generic search results
                return nba_results
        
        try:
            from urllib.parse import quote
            # Enhanced search query for better results
            enhanced_query = query
            if 'today' not in query_lower and 'now' not in query_lower:
                if any(term in query_lower for term in ['nba', 'game', 'playing', 'president', 'news']):
                    enhanced_query = f"{query} today {datetime.now().strftime('%Y')}"
            
            # First try instant answers
            ddg_url = f"https://api.duckduckgo.com/?q={quote(enhanced_query)}&format=json&no_html=1"
            response = requests.get(ddg_url, timeout=5)
            
            results = []
            if response.status_code == 200:
                data = response.json()
                
                # Check various answer types
                if data.get('AbstractText'):
                    results.append(f"Title:Summary\nSnippet:{data['AbstractText']}\nLink:{data.get('AbstractURL', '')}")
                
                if data.get('Answer'):
                    results.append(f"Title:Quick Answer\nSnippet:{data['Answer']}\nLink:")
                
                if data.get('Definition'):
                    results.append(f"Title:Definition\nSnippet:{data['Definition']}\nLink:{data.get('DefinitionURL', '')}")
                
                # Get instant answer type results
                if data.get('Infobox'):
                    info = data['Infobox']
                    if 'content' in info:
                        for item in info['content'][:3]:
                            if 'label' in item and 'value' in item:
                                results.append(f"Title:{item['label']}\nSnippet:{item['value']}\nLink:")
                
                # Get related topics
                for topic in data.get('RelatedTopics', [])[:3]:
                    if isinstance(topic, dict) and topic.get('Text'):
                        title = topic.get('Text', '').split(' - ')[0][:50]
                        results.append(f"Title:{title}\nSnippet:{topic['Text']}\nLink:{topic.get('FirstURL', '')}")
                
                if results:
                    return "\n\n".join(results)
            
            # Try HTML search with better parsing
            headers = {'User-Agent': self.user_agent}
            html_url = f"https://html.duckduckgo.com/html/?q={quote(enhanced_query)}"
            response = requests.get(html_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                # Extract results using BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find all result blocks
                result_blocks = soup.find_all('div', class_=['result', 'web-result'])
                if not result_blocks:
                    # Try alternative selectors
                    result_blocks = soup.find_all('div', class_='links_main')
                
                for result in result_blocks[:7]:  # Get more results
                    try:
                        # Extract title and URL
                        title_elem = result.find('a', class_='result__a')
                        if not title_elem:
                            title_elem = result.find('a')
                        
                        if not title_elem:
                            continue
                            
                        title = title_elem.text.strip()
                        url = title_elem.get('href', '')
                        
                        # Extract snippet
                        snippet_elem = result.find('a', class_='result__snippet')
                        if not snippet_elem:
                            snippet_elem = result.find('span', class_='result__snippet')
                        if not snippet_elem:
                            snippet_elem = result.find(text=True, recursive=True)
                            
                        snippet = snippet_elem.text.strip() if hasattr(snippet_elem, 'text') else str(snippet_elem).strip()
                        
                        # Clean up snippet
                        if len(snippet) > 200:
                            snippet = snippet[:200] + "..."
                        
                        # Handle DuckDuckGo's redirect URLs
                        if url.startswith('//duckduckgo.com/l/?uddg='):
                            try:
                                from urllib.parse import unquote
                                url = unquote(url.split('uddg=')[1].split('&')[0])
                            except:
                                pass
                        
                        if title and snippet:
                            results.append(f"Title:{title}\nSnippet:{snippet}\nLink:{url}")
                    except Exception as e:
                        logger.warning(f"Error parsing result: {e}")
                        continue
                
                if results:
                    return "\n\n".join(results)
                    
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
        
        return "Search results are limited. For the most current information, please check official sources or news websites directly."
    
    def _get_nba_scores(self) -> str:
        """Get NBA scores from ESPN API"""
        try:
            # ESPN API endpoint for NBA scores
            today = datetime.now().strftime('%Y%m%d')
            espn_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={today}"
            
            response = requests.get(espn_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                events = data.get('events', [])
                
                if not events:
                    return f"Title:NBA Games Today\nSnippet:No NBA games scheduled for today ({datetime.now().strftime('%B %d, %Y')})\nLink:https://www.espn.com/nba/schedule"
                
                results = []
                results.append(f"Title:NBA Games Today ({datetime.now().strftime('%B %d, %Y')})\nSnippet:Live scores and schedule\nLink:https://www.espn.com/nba/scoreboard")
                
                for event in events[:5]:
                    competition = event.get('competitions', [{}])[0]
                    competitors = competition.get('competitors', [])
                    
                    if len(competitors) >= 2:
                        away_team = competitors[1]
                        home_team = competitors[0]
                        
                        away_name = away_team.get('team', {}).get('displayName', 'Unknown')
                        home_name = home_team.get('team', {}).get('displayName', 'Unknown')
                        away_score = away_team.get('score', '0')
                        home_score = home_team.get('score', '0')
                        
                        status = competition.get('status', {})
                        game_status = status.get('type', {}).get('description', 'Scheduled')
                        
                        # Get more detailed status
                        period = status.get('period', 0)
                        clock = status.get('displayClock', '')
                        completed = status.get('type', {}).get('completed', False)
                        
                        if game_status == 'Scheduled':
                            start_time = event.get('date', '')
                            try:
                                from datetime import datetime
                                game_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                                game_time_str = game_time.strftime('%I:%M %p ET')
                                snippet = f"{away_name} @ {home_name} - Starts at {game_time_str}"
                            except:
                                snippet = f"{away_name} @ {home_name} - Scheduled"
                        elif completed:
                            snippet = f"{away_name} {away_score} - {home_name} {home_score} (Final)"
                        elif game_status == 'In Progress':
                            quarter = f"Q{period}" if period <= 4 else f"OT{period-4}"
                            snippet = f"{away_name} {away_score} - {home_name} {home_score} ({quarter} {clock})"
                        else:
                            snippet = f"{away_name} {away_score} - {home_name} {home_score} ({game_status})"
                        
                        # Add series info if playoffs
                        series = event.get('series', {})
                        if series:
                            series_summary = series.get('summary', '')
                            if series_summary:
                                snippet += f" | {series_summary}"
                        
                        results.append(f"Title:NBA Game\nSnippet:{snippet}\nLink:https://www.espn.com/nba/game/_/gameId/{event.get('id', '')}")
                
                return "\n\n".join(results)
                
        except Exception as e:
            logger.error(f"NBA scores error: {e}")
        
        return ""

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./yappy.db")
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

# Agent Router (matching local app's logic)
class AgentRouter:
    """Routes queries to appropriate agents based on local app's router.py"""
    
    def route_query(self, query: str) -> Tuple[str, bool]:
        """
        Route query to appropriate agent using pattern matching
        Returns: (agent_type, needs_search)
        """
        query_lower = query.lower()
        
        # Code agent patterns - check first for programming keywords
        code_patterns = [
            'code', 'script', 'program', 'function', 'debug', 'error', 'bug',
            'python', 'javascript', 'java', 'c++', 'c#', 'ruby', 'go', 'rust',
            'html', 'css', 'sql', 'bash', 'shell', 'powershell',
            'algorithm', 'implement', 'fix', 'syntax', 'compile', 'run',
            'class', 'method', 'variable', 'loop', 'array', 'api',
            'framework', 'library', 'package', 'module', 'import',
            'test', 'unit test', 'testing', 'exception', 'stack trace'
        ]
        
        # File agent patterns - file system operations
        file_patterns = [
            'file', 'folder', 'directory', 'path', 'document',
            'create file', 'read file', 'write file', 'delete file',
            'find file', 'locate', 'search file', 'list files',
            'move file', 'copy file', 'rename file', 'backup',
            'csv', 'txt', 'json', 'xml', 'pdf', 'doc',
            'download', 'upload', 'save', 'open file'
        ]
        
        # Planner agent patterns - planning and organizing
        # Planner agent patterns - task decomposition and project planning ONLY
        # NOT for travel planning (that goes to Casual Agent)
        planner_patterns = [
            'build a web app', 'create an application', 'develop a system',
            'make a program', 'setup a project', 'implement a solution',
            'organize files', 'structure a project', 'design a system',
            'coordinate tasks', 'break down project', 'decompose task',
            'project roadmap', 'development strategy', 'implementation plan',
            'multi-step project', 'complex task', 'workflow design'
        ]
        
        # Research agent patterns - deep learning and research
        research_patterns = [
            'research', 'study', 'learn about', 'understand', 'explain',
            'how does', 'why does', 'what causes', 'theory', 'concept',
            'tutorial', 'guide', 'lesson', 'teach', 'education',
            'science', 'history', 'mathematics', 'physics', 'chemistry',
            'biology', 'philosophy', 'psychology', 'sociology',
            'analysis', 'deep dive', 'comprehensive', 'detailed explanation'
        ]
        
        # Travel agent patterns - travel planning and tourism
        travel_patterns = [
            'travel', 'trip', 'vacation', 'holiday', 'tour', 'visit',
            'itinerary', 'destination', 'tourist', 'tourism', 'sightseeing',
            'flight', 'hotel', 'accommodation', 'airbnb', 'booking',
            'passport', 'visa', 'customs', 'airport', 'luggage',
            'beach', 'mountain', 'city break', 'weekend getaway',
            'backpack', 'resort', 'cruise', 'road trip', 'journey',
            'plan to go', 'planning to visit', 'want to travel', 'help me plan',
            'plan a trip', 'need to plan a trip', 'planning a trip'
        ]
        
        # Browser agent patterns - web search needed
        browser_patterns = [
            # Direct web requests
            'search', 'google', 'find online', 'look up', 'browse', 'web', 'internet',
            # Information queries
            'who is', 'what is', 'when is', 'where is', 'how much', 'how many',
            'latest', 'current', 'recent', 'news', 'update', 'trending',
            # Specific topics
            'weather', 'temperature', 'forecast', 'climate',
            'price', 'cost', 'stock', 'market', 'crypto', 'bitcoin',
            'president', 'ceo', 'company', 'organization', 'business',
            'game', 'score', 'sports', 'nba', 'nfl', 'soccer', 'baseball',
            # Research
            'research', 'article', 'paper', 'study', 'statistics',
            'information about', 'tell me about', 'facts about', 'learn about',
            'wikipedia', 'definition', 'meaning', 'explain what'
        ]
        
        # Priority-based routing (order matters!)
        # 1. Check for travel patterns first (high priority)
        if any(pattern in query_lower for pattern in travel_patterns):
            logger.info(f"Travel pattern matched for query: '{query_lower}'")
            return "travel", True
            
        # 2. Check for code patterns (most specific)
        elif any(pattern in query_lower for pattern in code_patterns):
            # Double check it's not a web search for code
            if any(web_word in query_lower for web_word in ['search online', 'find online', 'google', 'web']):
                return "browser", True
            return "code", False
            
        # 3. Check for file operations
        elif any(pattern in query_lower for pattern in file_patterns):
            return "file", False
            
        # 4. Check for planning/organizing
        elif any(pattern in query_lower for pattern in planner_patterns):
            return "planner", False
            
        # 5. Check for research/learning (but not web search)
        elif any(pattern in query_lower for pattern in research_patterns):
            # If it needs current info, use browser
            if any(web_word in query_lower for web_word in ['latest', 'current', 'recent', 'news']):
                return "browser", True
            return "research", False
            
        # 6. Check for web search needs
        elif any(pattern in query_lower for pattern in browser_patterns):
            return "browser", True
            
        # 7. Default to casual for general conversation
        else:
            return "casual", False

# Helper function to override model to OpenAI
def override_to_openai(model_name: str, api_key: str, llm_handler) -> Tuple[str, str]:
    """Override model to use OpenAI for all agents except Code Agent"""
    if model_name != "openai":
        logger.info(f"Overriding model from {model_name} to OpenAI")
        model_name = "openai"
        # Always use environment variable for security
        api_key = os.getenv('OPENAI_API_KEY')
    return model_name, api_key

# Browser Agent Implementation
class BrowserAgent:
    """Browser agent with web search capabilities"""
    
    def __init__(self):
        self.searx_tool = ImprovedSearch()
        self.date = datetime.now().strftime("%B %d, %Y")
        
    async def process(self, query: str, llm_handler, api_key: str, model_name: str, conversation_history: List[Dict] = None) -> str:
        """Process query with web search"""
        
        # Override to use OpenAI for Browser Agent
        model_name, api_key = override_to_openai(model_name, api_key, llm_handler)
        
        # Perform web search
        logger.info(f"Browser agent searching for: {query}")
        search_results = self.searx_tool.execute(query)
        
        # Build context for LLM
        system_prompt = f"""You are an AI assistant with web browsing capabilities.
Today's date is {self.date}.
You have searched the web and found the following results.
Use these search results to provide accurate, current information.
Be friendly and conversational."""
        
        user_prompt = f"""User question: {query}

Web search results:
{search_results}

IMPORTANT: You MUST use the search results above to answer the user's question with specific, factual information.
- If the results show NBA games, list the specific teams and scores/times
- If the results show weather data, provide the exact temperature and conditions
- If the results show news or current events, cite the specific information found
- Always be specific and use the actual data from the search results
- Do NOT give generic responses like "check ESPN" - instead, tell them what you found

Based on the search results above, provide a specific, detailed answer using the actual information found."""
        
        # Get response from LLM
        return await llm_handler._call_llm(system_prompt, user_prompt, model_name, api_key, conversation_history)

# Planner Agent Implementation
class PlannerAgent:
    """Planning agent for trips, schedules, and organizing tasks"""
    
    def __init__(self):
        self.searx_tool = ImprovedSearch()
        self.date = datetime.now().strftime("%B %d, %Y")
    
    async def process(self, query: str, llm_handler, api_key: str, model_name: str, conversation_history: List[Dict] = None) -> str:
        """Process planning requests with web search for current information"""
        
        # Override to use OpenAI for Planner Agent
        model_name, api_key = override_to_openai(model_name, api_key, llm_handler)
        
        # Check if query needs current information
        needs_search = any(word in query.lower() for word in ['price', 'cost', 'weather', 'available', 'open', 'current', 'latest', 'now'])
        
        search_results = ""
        if needs_search:
            logger.info(f"Planner agent searching for current info: {query}")
            search_results = self.searx_tool.execute(query)
        
        system_prompt = f"""You are an AI assistant specialized in planning and organizing!
You help users plan trips, create itineraries, organize schedules, and arrange activities.
Be detailed and ask clarifying questions when needed.
Today's date is {self.date}.

When helping with trip planning:
1. Ask about destination, dates, budget, and preferences
2. Suggest activities, accommodations, and transportation
3. Create detailed itineraries
4. Provide practical tips and recommendations
5. Use current information from web searches when available"""
        
        if search_results:
            user_prompt = f"""User request: {query}

Current information from web search:
{search_results}

Use the search results above to provide accurate, up-to-date information about prices, availability, weather, etc."""
        else:
            user_prompt = f"User request: {query}"
        
        # Get response from LLM
        return await llm_handler._call_llm(system_prompt, user_prompt, model_name, api_key, conversation_history)

# Code Agent Implementation
class CodeAgent:
    """Code generation and debugging agent with web search for documentation"""
    
    def __init__(self):
        self.searx_tool = ImprovedSearch()
        self.date = datetime.now().strftime("%B %d, %Y")
    
    async def process(self, query: str, llm_handler, api_key: str, model_name: str, conversation_history: List[Dict] = None) -> str:
        """Process coding requests - Always uses Claude/Anthropic"""
        
        # ALWAYS use Claude for Code Agent
        claude_api_key = os.getenv('CLAUDE_API_KEY')
        if claude_api_key:
            model_name = "anthropic"
            api_key = claude_api_key
            logger.info("Code Agent: Using Claude API for enhanced coding assistance")
        
        # Check if query needs documentation lookup
        needs_docs = any(word in query.lower() for word in [
            'how to', 'documentation', 'docs', 'api', 'library', 'framework',
            'latest version', 'changelog', 'deprecated', 'best practice'
        ])
        
        search_results = ""
        if needs_docs:
            logger.info(f"Code agent searching for documentation: {query}")
            search_results = self.searx_tool.execute(query)
        
        system_prompt = f"""You are an AI assistant specialized in coding and programming!
You help users write code, debug programs, explain algorithms, and solve technical problems.
Today's date is {self.date}.
Be helpful and clear in your explanations.

When helping with code:
1. Write clean, well-commented code
2. Explain your approach clearly
3. Suggest best practices and improvements
4. Help debug errors with patience
5. Use current documentation when available"""
        
        if search_results:
            user_prompt = f"""User coding request: {query}

Current documentation/information from web:
{search_results}

Use the search results to provide accurate, up-to-date code examples and documentation references."""
        else:
            user_prompt = f"User coding request: {query}"
        
        # Get response from LLM
        return await llm_handler._call_llm(system_prompt, user_prompt, model_name, api_key, conversation_history)

# Research Agent Implementation
class ResearchAgent:
    """Research and learning agent with web search"""
    
    def __init__(self):
        self.searx_tool = ImprovedSearch()
        self.date = datetime.now().strftime("%B %d, %Y")
    
    async def process(self, query: str, llm_handler, api_key: str, model_name: str, conversation_history: List[Dict] = None) -> str:
        """Process research and learning requests with web search for current information"""
        
        # Override to use OpenAI for Research Agent
        model_name, api_key = override_to_openai(model_name, api_key, llm_handler)
        
        # Always search for research topics to get latest information
        logger.info(f"Research agent searching for: {query}")
        search_results = self.searx_tool.execute(query)
        
        system_prompt = f"""You are an AI assistant specialized in research and learning!
You help users understand complex topics, conduct research, and learn new things.
Be thorough, accurate, and educational.
Today's date is {self.date}.

When helping with research:
1. Break down complex topics into understandable parts
2. Provide accurate and well-sourced information from search results
3. Explain concepts clearly with examples
4. Suggest additional resources for learning
5. Always use the most current information available"""
        
        user_prompt = f"""User research request: {query}

Current research from web search:
{search_results}

Use the search results above to provide comprehensive, accurate, and up-to-date information on this topic."""
        
        # Get response from LLM
        return await llm_handler._call_llm(system_prompt, user_prompt, model_name, api_key, conversation_history)

# File Agent Implementation
class FileAgent:
    """File and directory management agent with web search for tutorials"""
    
    def __init__(self):
        self.searx_tool = ImprovedSearch()
        self.date = datetime.now().strftime("%B %d, %Y")
    
    async def process(self, query: str, llm_handler, api_key: str, model_name: str, conversation_history: List[Dict] = None) -> str:
        """Process file system requests"""
        
        # Override to use OpenAI for File Agent
        model_name, api_key = override_to_openai(model_name, api_key, llm_handler)
        
        # Check if query needs tutorial/guide lookup
        needs_tutorial = any(word in query.lower() for word in [
            'how to', 'tutorial', 'guide', 'best way', 'organize', 'structure',
            'backup', 'sync', 'cloud storage', 'file format'
        ])
        
        search_results = ""
        if needs_tutorial:
            logger.info(f"File agent searching for tutorials: {query}")
            search_results = self.searx_tool.execute(query)
        
        system_prompt = f"""You are an AI assistant specialized in file and directory management!
You help users find files, organize directories, and manage their file system.
Today's date is {self.date}.
Be helpful and provide clear instructions.

When helping with files:
1. Provide clear file paths and commands
2. Explain file operations step by step
3. Suggest organization strategies
4. Help with file searches and management
5. Use current best practices when available"""
        
        if search_results:
            user_prompt = f"""User file request: {query}

Current information/tutorials from web:
{search_results}

Use the search results to provide accurate, up-to-date guidance on file management."""
        else:
            user_prompt = f"User file request: {query}"
        
        # Get response from LLM
        return await llm_handler._call_llm(system_prompt, user_prompt, model_name, api_key, conversation_history)

# Casual Agent Implementation
class CasualAgent:
    """Casual conversation agent with optional web search"""
    
    def __init__(self):
        self.searx_tool = ImprovedSearch()
        self.date = datetime.now().strftime("%B %d, %Y")
    
    async def process(self, query: str, llm_handler, api_key: str, model_name: str, conversation_history: List[Dict] = None) -> str:
        """Process casual conversation with web search for factual questions"""
        
        # Override to use OpenAI for Casual Agent
        model_name, api_key = override_to_openai(model_name, api_key, llm_handler)
        
        # Check if query needs factual/current information
        needs_search = any(word in query.lower() for word in ['who', 'what', 'when', 'where', 'how many', 'how much', 'news', 'current', 'latest'])
        
        search_results = ""
        if needs_search:
            logger.info(f"Casual agent searching for factual info: {query}")
            search_results = self.searx_tool.execute(query)
        
        system_prompt = f"""You are a friendly and helpful AI assistant!
Be cheerful and helpful.
Always aim to brighten the user's day with your positive energy!
Today's date is {self.date}.

When web search results are available, use them to provide accurate, current information."""
        
        if search_results:
            user_prompt = f"""User: {query}

Current information from web:
{search_results}

Respond in a friendly, casual way while incorporating any relevant facts from the search results."""
        else:
            user_prompt = query
        
        return await llm_handler._call_llm(system_prompt, user_prompt, model_name, api_key, conversation_history)

# Travel Agent Implementation
class TravelAgent:
    """Travel planning agent with web search for destinations, flights, hotels, and activities"""
    
    def __init__(self):
        self.searx_tool = ImprovedSearch()
        self.date = datetime.now().strftime("%B %d, %Y")
    
    async def process(self, query: str, llm_handler, api_key: str, model_name: str, conversation_history: List[Dict] = None) -> str:
        """Process travel planning queries with web search"""
        
        # Override to use OpenAI for Travel Agent
        model_name, api_key = override_to_openai(model_name, api_key, llm_handler)
        
        # Always search for travel information
        logger.info(f"Travel agent searching for: {query}")
        search_results = self.searx_tool.execute(query)
        
        system_prompt = f"""You are an expert travel planner with extensive knowledge of destinations worldwide.
Today's date is {self.date}.
You help users plan amazing trips by providing detailed itineraries, recommendations, and travel tips.
You always search for current information about destinations, flights, hotels, weather, and local attractions.

When creating travel plans, include:
- Day-by-day itineraries with specific activities and timings
- Accommodation recommendations with price ranges
- Transportation options (flights, trains, local transport)
- Must-see attractions and hidden gems
- Restaurant and dining recommendations
- Budget estimates for different travel styles
- Weather considerations and what to pack
- Local customs and travel tips
- Safety information if relevant

Be enthusiastic about travel and help users get excited about their upcoming trips!"""
        
        user_prompt = f"""User's travel request: {query}

Current travel information from web search:
{search_results}

Create a comprehensive travel plan using the search results above. Be specific with recommendations and include practical details like costs, timings, and booking tips."""
        
        return await llm_handler._call_llm(system_prompt, user_prompt, model_name, api_key, conversation_history)

# LLM Handler
class LLMHandler:
    """Handles LLM interactions"""
    
    def __init__(self):
        self.router = AgentRouter()
        # Initialize agents (some now have web search capabilities)
        self.agents = {
            "browser": BrowserAgent(),      # Already has web search
            "casual": CasualAgent(),        # Now has optional web search
            "planner": PlannerAgent(),      # Now has web search for current info
            "code": CodeAgent(),            # Now has web search for docs
            "file": FileAgent(),            # Now has web search for tutorials
            "research": ResearchAgent(),    # Now always uses web search
            "travel": TravelAgent()         # Travel planning with web search
        }
        
    async def get_response(self, prompt: str, model_name: str, api_key: Optional[str] = None, 
                          conversation_history: List[Dict] = None, file_data: Optional[Dict] = None) -> Tuple[str, int]:
        """Get response using appropriate agent"""
        
        # If there's file data, process it appropriately
        if file_data:
            # Add file context to prompt
            file_info = f"\n\n[User uploaded file: {file_data['name']} (type: {file_data['type']})]"
            logger.info(f"Processing file upload: {file_data['name']} ({file_data['type']})")
            
            # For images, we need to use a vision-capable model
            if file_data['type'].startswith('image/'):
                logger.info(f"Handling image with model: {model_name}, has_api_key: {bool(api_key)}")
                return await self._handle_image(prompt, file_data, model_name, api_key, conversation_history)
            else:
                # For text files, include content in the prompt
                prompt = f"{prompt}{file_info}\n\nFile content:\n{file_data['content'][:5000]}"  # Limit content length
        
        # Route to appropriate agent
        agent_type, needs_search = self.router.route_query(prompt)
        logger.info(f"Routing to {agent_type} agent, needs_search: {needs_search}")
        
        # Get agent
        agent = self.agents.get(agent_type, self.agents["casual"])
        
        # Generate routing message
        routing_message = ""
        agent_names = {
            "browser": "Browser Agent 🌐",
            "planner": "Planning Agent 📅",
            "code": "Code Agent 💻",
            "file": "File Agent 📁",
            "research": "Research Agent 🔬",
            "casual": "Casual Agent",
            "travel": "Travel Agent ✈️"
        }
        
        if conversation_history:
            # Check last agent used
            last_agent = None
            for msg in reversed(conversation_history):
                if isinstance(msg, dict) and 'agent_used' in msg:
                    last_agent = msg.get('agent_used')
                    break
            
            # Show routing message if switching agents (not to casual)
            if last_agent and last_agent != agent_type and agent_type != "casual":
                routing_message = f"\n\n*[Routing you to {agent_names.get(agent_type, agent_type.title() + ' Agent')} for specialized assistance...]*\n\n"
        else:
            # First message - show which agent is handling if not casual
            if agent_type != "casual":
                routing_message = f"\n\n*[{agent_names.get(agent_type, agent_type.title() + ' Agent')} is here to help!]*\n\n"
        
        # Process with agent
        logger.info(f"Processing with {agent_type} agent, api_key present: {bool(api_key)}")
        response = await agent.process(prompt, self, api_key, model_name, conversation_history)
        
        # Add routing message to response if applicable
        if routing_message:
            response = routing_message + response
        
        # Special handling for API key errors to preserve routing message
        if "I need an API key" in response and routing_message:
            # Ensure routing message is visible even with API key error
            response = routing_message + "\n" + response
        
        return response, 0  # Token count would be calculated by LLM
    
    async def _handle_image(self, prompt: str, file_data: Dict, model_name: str, 
                           api_key: Optional[str], conversation_history: List[Dict] = None) -> Tuple[str, int]:
        """Handle image analysis using vision models"""
        
        if not api_key:
            logger.warning(f"No API key found for model: {model_name}")
            return f"Server configuration error: No API key configured for {model_name}. Please contact administrator.", 0
        
        try:
            # For OpenAI, use GPT-4 Vision
            if model_name == "openai" and openai:
                client = openai.OpenAI(api_key=api_key)
                
                messages = [
                    {
                        "role": "system",
                        "content": "You are an AI assistant that can analyze images! Be helpful and informative."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": file_data['content']  # Base64 image data
                                }
                            }
                        ]
                    }
                ]
                
                response = client.chat.completions.create(
                    model="gpt-4o",  # Updated to use gpt-4o which supports vision
                    messages=messages,
                    max_tokens=1000
                )
                
                return response.choices[0].message.content, 0
            
            # For Anthropic Claude
            elif model_name == "anthropic" and anthropic:
                client = anthropic.Anthropic(api_key=api_key)
                
                # Extract base64 data from data URL
                import base64
                base64_data = file_data['content'].split(',')[1] if ',' in file_data['content'] else file_data['content']
                
                response = client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=1000,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"You are an AI assistant! {prompt}"
                                },
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": file_data['type'],
                                        "data": base64_data
                                    }
                                }
                            ]
                        }
                    ]
                )
                
                return response.content[0].text, 0
            
            # For Google Gemini
            elif (model_name == "google" or model_name == "gemini") and genai and PIL_AVAILABLE:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-pro-vision')
                
                # Convert base64 to PIL Image
                import base64
                
                base64_data = file_data['content'].split(',')[1] if ',' in file_data['content'] else file_data['content']
                image_data = base64.b64decode(base64_data)
                image = Image.open(io.BytesIO(image_data))
                
                response = model.generate_content([
                    f"You are an AI assistant! {prompt}",
                    image
                ])
                
                return response.text, 0
            
            else:
                # For models without vision capability, provide helpful alternatives
                response = f"""I see you've uploaded an image ({file_data['name']}), but {model_name} doesn't have built-in vision capabilities. 

Here's how you can get your image analyzed:

**Option 1: Use a Vision-Capable Model**
• Switch to OpenAI (with GPT-4 Vision)
• Switch to Anthropic (with Claude 3) 
• Switch to Google (with Gemini Pro Vision)

**Option 2: Use Free Online Tools**
Here are some popular AI image analysis websites:

• **AI Describe Image | ImagePrompt.org**
  - Provides detailed descriptions and object identification
  - Can answer specific questions about your image

• **Image Describer - YesChat**
  - Offers descriptions in text and table formats
  - Advanced image analysis features

• **Image Explainer - AIChatOnline.org**
  - Uses advanced image recognition
  - Provides easy-to-understand descriptions

**Option 3: Describe It to Me**
Tell me what's in the image and I'll help you with any questions about it!

Which option would you like to try?"""
                
                return response, 0
            
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            error_msg = str(e)
            
            # Provide more helpful error messages
            if "model does not exist" in error_msg.lower():
                return f"The vision model isn't available. Make sure you're using a valid OpenAI API key with access to GPT-4 vision models.", 0
            elif "api key" in error_msg.lower():
                return f"There's an issue with your API key. Please check that it's valid and has the necessary permissions.", 0
            elif "rate limit" in error_msg.lower():
                return f"You've hit the rate limit. Please wait a moment and try again.", 0
            else:
                return f"I encountered an error analyzing the image: {error_msg}", 0
    
    async def _call_llm(self, system_prompt: str, user_prompt: str, model_name: str, 
                        api_key: Optional[str], conversation_history: List[Dict] = None) -> str:
        """Call the appropriate LLM"""
        
        if not api_key:
            return f"Server configuration error: No API key configured for {model_name}. Please contact administrator."
        
        try:
            if model_name == "openai":
                if not openai:
                    logger.error("OpenAI library not available - attempting to import")
                    try:
                        import openai as openai_module
                        openai = openai_module
                    except ImportError as e:
                        logger.error(f"Failed to import openai: {e}")
                        return "OpenAI library is not installed. Please ensure 'openai' is in requirements.txt and redeploy."
                
                client = openai.OpenAI(api_key=api_key)
                
                messages = [{"role": "system", "content": system_prompt}]
                
                if conversation_history:
                    for msg in conversation_history[-5:]:
                        messages.append({"role": "user", "content": msg.get("user_message", "")})
                        messages.append({"role": "assistant", "content": msg.get("assistant_response", "")})
                
                messages.append({"role": "user", "content": user_prompt})
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # Use GPT-4o-mini model (fast and cost-effective)
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                return response.choices[0].message.content
            
            elif model_name == "anthropic" and anthropic:
                client = anthropic.Anthropic(api_key=api_key)
                
                messages = []
                if conversation_history:
                    for msg in conversation_history[-5:]:
                        messages.append({"role": "user", "content": msg.get("user_message", "")})
                        messages.append({"role": "assistant", "content": msg.get("assistant_response", "")})
                
                messages.append({"role": "user", "content": user_prompt})
                
                response = client.messages.create(
                    model="claude-3-sonnet-20240229",
                    system=system_prompt,
                    messages=messages,
                    max_tokens=1000
                )
                
                return response.content[0].text
            
            elif (model_name == "google" or model_name == "gemini") and genai:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                full_prompt = system_prompt + "\n\n"
                if conversation_history:
                    for msg in conversation_history[-5:]:
                        full_prompt += f"User: {msg.get('user_message', '')}\n"
                        full_prompt += f"Assistant: {msg.get('assistant_response', '')}\n"
                
                full_prompt += f"User: {user_prompt}\nAssistant:"
                
                response = model.generate_content(full_prompt)
                return response.text
            
            elif model_name == "groq" and Groq:
                client = Groq(api_key=api_key)
                
                messages = [{"role": "system", "content": system_prompt}]
                if conversation_history:
                    for msg in conversation_history[-5:]:
                        messages.append({"role": "user", "content": msg.get("user_message", "")})
                        messages.append({"role": "assistant", "content": msg.get("assistant_response", "")})
                
                messages.append({"role": "user", "content": user_prompt})
                
                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                return response.choices[0].message.content
            
            else:
                return f"The {model_name} model isn't available. Try another one!"
                
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Model: {model_name}, Has API key: {bool(api_key)}")
            
            # More specific error messages
            error_str = str(e).lower()
            if "connection" in error_str:
                return "Connection error: Unable to reach the AI service. Please check your internet connection."
            elif "401" in error_str or "unauthorized" in error_str:
                return "Authentication error: Your API key may be invalid or expired."
            elif "429" in error_str or "rate limit" in error_str:
                return "Rate limit exceeded: Please wait a moment and try again."
            elif "timeout" in error_str:
                return "Request timed out. Please try again."
            else:
                return f"I encountered an error: {str(e)}"

# FastAPI app setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Yappy AI (Fixed Agent Routing) is starting up...")
    print(f"Current date: {datetime.now().strftime('%A, %B %d, %Y')}")
    
    try:
        await database.connect()
        print("✅ Database connected successfully")
        
        engine = sqlalchemy.create_engine(DATABASE_URL)
        metadata.create_all(bind=engine)
        print("✅ Database tables created/verified")
        
        # Create default admin user if it doesn't exist
        query = users_table.select().where(users_table.c.username == "admin")
        existing_user = await database.fetch_one(query)
        
        if not existing_user:
            salt = secrets.token_hex(16)
            password_hash = hash_password("yappy123", salt)
            
            insert_query = users_table.insert().values(
                username="admin",
                email="admin@yappy.ai",
                password_hash=password_hash,
                salt=salt,
                created_at=datetime.utcnow()
            )
            await database.execute(insert_query)
            print("✅ Default admin user created (username: admin, password: yappy123)")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
    
    yield
    
    # Shutdown
    print("Yappy AI is shutting down... Goodbye!")
    if database.is_connected:
        await database.disconnect()

app = FastAPI(title="Yappy AI Assistant", version="7.0.0", lifespan=lifespan)

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

class FileData(BaseModel):
    name: str
    type: str
    content: str

class ChatRequest(BaseModel):
    message: str
    model_name: Optional[str] = "openai"
    conversation_id: Optional[str] = None
    stream: Optional[bool] = False
    file_data: Optional[FileData] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    message_id: str
    timestamp: str
    model_used: str
    tokens_used: Optional[int] = None
    agent_used: Optional[str] = None

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
        
        if not database.is_connected:
            return username
        
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        if not user:
            return username
        
        return username
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        try:
            return credentials.credentials.split(":")[0]
        except:
            raise HTTPException(status_code=401, detail="Invalid token")

# Initialize components
llm_handler = LLMHandler()

# Import enhanced handler if available
try:
    from sources.llm_handler_v2 import EnhancedLLMHandler
    enhanced_handler = EnhancedLLMHandler()
    MULTI_AGENT_ENABLED = True
except ImportError:
    # Try minimal fallback handler
    try:
        from sources.llm_handler_minimal import EnhancedLLMHandler
        enhanced_handler = EnhancedLLMHandler()
        MULTI_AGENT_ENABLED = True
        logger.info("Using minimal multi-agent handler")
    except ImportError:
        enhanced_handler = None
        MULTI_AGENT_ENABLED = False
        logger.warning("Enhanced multi-agent system not available")

# API Endpoints
@app.get("/")
async def root(request: Request):
    """Serve unified responsive chat interface"""
    # First try to serve the main index.html (with latest updates)
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    
    # Fallback to claude UI version
    claude_ui_path = os.path.join(static_dir, "yappy_claude_ui.html")
    if os.path.exists(claude_ui_path):
        return FileResponse(claude_ui_path)
    
    # Fallback to responsive version
    responsive_path = os.path.join(static_dir, "yappy_responsive.html")
    if os.path.exists(responsive_path):
        return FileResponse(responsive_path)
    
    # Fallback to device-specific versions if responsive doesn't exist
    user_agent = request.headers.get('user-agent', '').lower()
    
    # Check if mobile device
    is_mobile = any(device in user_agent for device in ['mobile', 'android', 'iphone', 'ipad', 'ipod'])
    
    if is_mobile:
        # Serve mobile-optimized version
        mobile_path = os.path.join(static_dir, "index_mobile.html")
        if os.path.exists(mobile_path):
            return FileResponse(mobile_path)
        # Fallback to yappy_mobile_v2.html if index_mobile doesn't exist
        mobile_v2_path = os.path.join(static_dir, "yappy_mobile_v2.html")
        if os.path.exists(mobile_v2_path):
            return FileResponse(mobile_v2_path)
    
    # Serve desktop version
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
        "version": "7.0.0",
        "current_date": datetime.now().strftime("%B %d, %Y"),
        "features": ["browser_agent", "casual_agent", "web_search"],
        "database": "connected" if database.is_connected else "disconnected"
    }

@app.post("/auth/register", response_model=TokenResponse)
async def register(user_data: UserSignup):
    """Register new user"""
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
        query = users_table.select().where(users_table.c.username == user_data.username)
        user = await database.fetch_one(query)
        
        if not user or user.password_hash != hash_password(user_data.password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
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
    """Main chat endpoint with proper agent routing"""
    try:
        # Get user
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get API key from environment variables only (server-side security)
        if request.model_name == "anthropic":
            api_key = os.getenv('CLAUDE_API_KEY')
        elif request.model_name == "openai":
            api_key = os.getenv('OPENAI_API_KEY')
        elif request.model_name == "google" or request.model_name == "gemini":
            api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        elif request.model_name == "groq":
            api_key = os.getenv('GROQ_API_KEY')
        else:
            api_key = None
        
        # Debug logging
        logger.info(f"User: {username}, Model: {request.model_name}")
        logger.info(f"API key found from environment: {'Yes' if api_key else 'No'}")
        if not api_key and request.model_name == "openai":
            logger.error(f"OpenAI key check - env var: {os.getenv('OPENAI_API_KEY')[:10] if os.getenv('OPENAI_API_KEY') else 'None'}")
        
        # If no API key found, return error
        if not api_key:
            logger.error(f"No API key configured on server for model: {request.model_name}")
            
            # Check if running on Render
            is_render = os.getenv('RENDER') == 'true'
            
            if is_render:
                detail = f"Server configuration error: No {request.model_name.upper()}_API_KEY environment variable found. Please add it in Render Dashboard > Environment."
            else:
                detail = f"Server configuration error: No API key configured for {request.model_name}. Please check your .env file or environment variables."
            
            raise HTTPException(
                status_code=500, 
                detail=detail
            )
        
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
        
        # Get response from appropriate agent
        file_data = None
        if request.file_data:
            file_data = {
                'name': request.file_data.name,
                'type': request.file_data.type,
                'content': request.file_data.content
            }
        
        # No need to pass user API keys since we only use server environment variables
        
        response_text, tokens = await llm_handler.get_response(
            request.message,
            request.model_name,
            api_key,
            conversation_messages,
            file_data
        )
        
        # Determine which agent was used
        agent_type, _ = llm_handler.router.route_query(request.message)
        
        # Store message
        message_id = str(uuid.uuid4())
        message_data = {
            "id": message_id,
            "user_message": request.message,
            "assistant_response": response_text,
            "model": request.model_name,
            "timestamp": datetime.now().isoformat(),
            "tokens": tokens,
            "agent_used": agent_type
        }
        
        # Add file info if present
        if request.file_data:
            message_data["file_attachment"] = {
                "name": request.file_data.name,
                "type": request.file_data.type
            }
        
        # Update conversation
        conversation_messages.append(message_data)
        
        # Handle missing updated_at column
        try:
            update_query = conversations_table.update().where(
                conversations_table.c.id == conv_id
            ).values(
                messages=conversation_messages,
                updated_at=datetime.now()
            )
            await database.execute(update_query)
        except Exception as e:
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
            agent_used=agent_type
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
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        api_keys = user.api_keys or {}
        
        # If the API key already exists for another model and this is the same key,
        # log a warning to help debug
        if update_data.api_key in api_keys.values():
            existing_model = [k for k, v in api_keys.items() if v == update_data.api_key][0]
            logger.info(f"Same API key being used for {update_data.model_name} (was already set for {existing_model})")
        
        api_keys[update_data.model_name] = update_data.api_key
        
        update_query = users_table.update().where(
            users_table.c.username == username
        ).values(api_keys=api_keys)
        
        await database.execute(update_query)
        
        logger.info(f"API key updated for user {username}, model {update_data.model_name}")
        return {"status": "success", "message": f"API key updated for {update_data.model_name}"}
        
    except Exception as e:
        logger.error(f"API key update error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update API key")

@app.post("/api/user/copy-api-key/{from_model}/{to_model}")
async def copy_api_key(from_model: str, to_model: str, username: str = Depends(verify_token)):
    """Copy API key from one model to another"""
    try:
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        api_keys = user.api_keys or {}
        
        if from_model not in api_keys:
            raise HTTPException(status_code=404, detail=f"No API key found for {from_model}")
        
        # Copy the API key
        api_keys[to_model] = api_keys[from_model]
        
        update_query = users_table.update().where(
            users_table.c.username == username
        ).values(api_keys=api_keys)
        
        await database.execute(update_query)
        
        logger.info(f"Copied API key from {from_model} to {to_model} for user {username}")
        return {"status": "success", "message": f"API key copied from {from_model} to {to_model}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Copy API key error: {e}")
        raise HTTPException(status_code=500, detail="Failed to copy API key")

@app.get("/api/user/api-keys")
async def get_user_api_keys(username: str = Depends(verify_token)):
    """Get list of models with API keys configured"""
    try:
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            return {"models": []}
        
        api_keys = user.api_keys or {}
        return {"models": list(api_keys.keys())}
        
    except Exception as e:
        logger.error(f"Get API keys error: {e}")
        return {"models": []}

@app.delete("/api/user/api-key/{model_name}")
async def delete_api_key(model_name: str, username: str = Depends(verify_token)):
    """Delete API key for a specific model"""
    try:
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        api_keys = user.api_keys or {}
        
        if model_name not in api_keys:
            raise HTTPException(status_code=404, detail=f"No API key found for {model_name}")
        
        # Remove the API key
        del api_keys[model_name]
        
        update_query = users_table.update().where(
            users_table.c.username == username
        ).values(api_keys=api_keys)
        
        await database.execute(update_query)
        
        logger.info(f"Deleted API key for {model_name} for user {username}")
        return {"status": "success", "message": f"API key for {model_name} removed"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete API key error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete API key")

@app.get("/api/conversations")
async def get_conversations(username: str = Depends(verify_token)):
    """Get user's conversations"""
    try:
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            return []
        
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
                "message_count": len(messages),
                "last_message": last_message.get("user_message") if last_message else None,
                "agent_used": last_message.get("agent_used", "casual") if last_message else None
            })
        
        return result
    except Exception as e:
        logger.error(f"Get conversations error: {e}")
        return []

@app.get("/api/conversation/{conversation_id}")
async def get_conversation(conversation_id: str, username: str = Depends(verify_token)):
    """Get specific conversation"""
    try:
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        conv_query = conversations_table.select().where(
            conversations_table.c.id == conversation_id
        )
        conversation = await database.fetch_one(conv_query)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
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

# Multi-Agent System Endpoints
@app.get("/api/agents", dependencies=[Depends(verify_token)])
async def get_agents(username: str = Depends(verify_token)):
    """Get available agents and their capabilities"""
    if not MULTI_AGENT_ENABLED:
        return {"error": "Multi-agent system not enabled"}
    
    return {
        "agents": enhanced_handler.get_capabilities(),
        "multi_agent_enabled": True
    }

@app.get("/api/agents/status", dependencies=[Depends(verify_token)])
async def get_agent_status(username: str = Depends(verify_token)):
    """Get real-time status of all agents"""
    if not MULTI_AGENT_ENABLED:
        return {"error": "Multi-agent system not enabled"}
    
    return enhanced_handler.get_agent_status()

@app.post("/api/tasks", dependencies=[Depends(verify_token)])
async def create_task(request: ChatRequest, username: str = Depends(verify_token)):
    """Create a new orchestrated task"""
    if not MULTI_AGENT_ENABLED:
        return {"error": "Multi-agent system not enabled"}
    
    try:
        # Get user for API keys
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        api_keys = user.api_keys or {}
        api_key = api_keys.get(request.model_name)
        
        # Create task through enhanced handler
        response, tokens = await enhanced_handler.get_response(
            request.message,
            request.model_name,
            api_key
        )
        
        # Extract task ID from response
        import re
        task_id_match = re.search(r'Task ID.*?`([^`]+)`', response)
        task_id = task_id_match.group(1) if task_id_match else None
        
        return {
            "task_id": task_id,
            "message": response,
            "status": "created"
        }
        
    except Exception as e:
        logger.error(f"Task creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tasks/{task_id}", dependencies=[Depends(verify_token)])
async def get_task_status(task_id: str, username: str = Depends(verify_token)):
    """Get status of a specific task"""
    if not MULTI_AGENT_ENABLED:
        return {"error": "Multi-agent system not enabled"}
    
    return enhanced_handler.get_task_status(task_id)

@app.post("/api/chat/multi-agent", dependencies=[Depends(verify_token)])
async def chat_multi_agent(request: ChatRequest, username: str = Depends(verify_token)):
    """Chat endpoint that uses the multi-agent system"""
    if not MULTI_AGENT_ENABLED:
        # Fallback to regular chat
        return await chat(request, username)
    
    try:
        # Get user and API keys
        query = users_table.select().where(users_table.c.username == username)
        user = await database.fetch_one(query)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        api_keys = user.api_keys or {}
        api_key = api_keys.get(request.model_name)
        
        # Process through enhanced handler
        response_text, tokens = await enhanced_handler.get_response(
            request.message,
            request.model_name,
            api_key,
            file_data=request.file_data.dict() if request.file_data else None
        )
        
        # Determine which agent was used
        agent_info = enhanced_handler.get_agent_status()
        
        return {
            "response": response_text,
            "model_used": request.model_name,
            "tokens_used": tokens,
            "agents_involved": agent_info.get("agents", {}),
            "multi_agent": True
        }
        
    except Exception as e:
        logger.error(f"Multi-agent chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)