"""
Improved search implementation with direct weather support
"""
import requests
import json
from typing import List, Dict, Any
import re
from urllib.parse import quote
import logging

logger = logging.getLogger(__name__)

class ImprovedSearch:
    def __init__(self):
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    
    def execute(self, query: str) -> str:
        """Execute search with smart routing"""
        query_lower = query.lower()
        
        # Check if it's a weather query
        if 'weather' in query_lower:
            return self._get_weather_data(query)
        
        # For other queries, try web search
        return self._web_search(query)
    
    def _get_weather_data(self, query: str) -> str:
        """Get weather data directly from wttr.in"""
        # Extract location from query
        location = self._extract_location(query)
        
        try:
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
        # Remove common words
        query = query.lower()
        query = query.replace("what's the", "").replace("what is the", "")
        query = query.replace("how's the", "").replace("how is the", "")
        query = query.replace("weather", "").replace("temperature", "")
        query = query.replace("forecast", "").replace("in", "").replace("at", "")
        query = query.replace("?", "").strip()
        
        # Clean up extra spaces
        location = " ".join(query.split())
        
        # Default to a location if empty
        if not location:
            location = "New York"
        
        return location
    
    def _web_search(self, query: str) -> str:
        """Perform general web search"""
        try:
            # Try Google search (using programmable search engine if available)
            google_results = self._google_search(query)
            if google_results:
                return google_results
        except Exception as e:
            logger.error(f"Google search error: {e}")
        
        try:
            # Try DuckDuckGo HTML search as fallback
            ddg_results = self._duckduckgo_search(query)
            if ddg_results:
                return ddg_results
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
        
        return "No search results found. Please try a different query or be more specific."
    
    def _google_search(self, query: str) -> str:
        """Try Google search using their JSON API"""
        # This would require a Google API key and Custom Search Engine ID
        # For now, skip this and use DuckDuckGo
        return ""
    
    def _duckduckgo_search(self, query: str) -> str:
        """Search using DuckDuckGo HTML"""
        try:
            # First try instant answers
            ddg_url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1"
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
                
                # Get related topics
                for topic in data.get('RelatedTopics', [])[:3]:
                    if isinstance(topic, dict) and topic.get('Text'):
                        title = topic.get('Text', '').split(' - ')[0][:50] + "..."
                        results.append(f"Title:{title}\nSnippet:{topic['Text']}\nLink:{topic.get('FirstURL', '')}")
                
                if results:
                    return "\n\n".join(results)
            
            # If no instant answers, try HTML parsing
            headers = {'User-Agent': self.user_agent}
            html_url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
            response = requests.get(html_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                # Simple extraction of results
                text = response.text
                
                # Extract results using regex
                import re
                # Look for result snippets
                result_pattern = r'<a class="result__a" href="([^"]+)".*?>([^<]+)</a>.*?<a class="result__snippet".*?>([^<]+)</a>'
                matches = re.findall(result_pattern, text, re.DOTALL)
                
                for url, title, snippet in matches[:5]:
                    # Clean up
                    title = title.strip()
                    snippet = snippet.strip()
                    # Handle DuckDuckGo's redirect URLs
                    if url.startswith('//duckduckgo.com/l/?uddg='):
                        try:
                            url = url.split('uddg=')[1].split('&')[0]
                            from urllib.parse import unquote
                            url = unquote(url)
                        except:
                            pass
                    
                    results.append(f"Title:{title}\nSnippet:{snippet}\nLink:{url}")
                
                if results:
                    return "\n\n".join(results)
                    
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
        
        return ""

# Test the search
if __name__ == "__main__":
    search = ImprovedSearch()
    
    # Test weather
    print("Testing weather search:")
    print(search.execute("what's the weather in garden grove, ca"))
    print("\n" + "="*50 + "\n")
    
    # Test general search
    print("Testing general search:")
    print(search.execute("who is the president of the united states"))