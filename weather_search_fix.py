import requests
import json
from urllib.parse import quote

class WeatherSearch:
    """Enhanced search with weather-specific capabilities"""
    
    def __init__(self):
        self.base_url = "https://search.brave.com/api/web"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def search_weather(self, query: str) -> str:
        """Search for weather information"""
        # Try OpenWeatherMap API (free tier)
        if "weather" in query.lower():
            # Extract location from query
            location = query.lower().replace("what's the weather in", "").replace("weather in", "").strip()
            location = location.replace("?", "").strip()
            
            # Try weather-specific APIs
            weather_results = self._get_weather_wttr(location)
            if weather_results:
                return weather_results
        
        # Fallback to web search
        return self._web_search(query)
    
    def _get_weather_wttr(self, location: str) -> str:
        """Get weather from wttr.in (free, no API key needed)"""
        try:
            # wttr.in provides free weather data
            url = f"https://wttr.in/{quote(location)}?format=j1"
            response = requests.get(url, headers=self.headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                current = data.get('current_condition', [{}])[0]
                
                if current:
                    temp_c = current.get('temp_C', 'N/A')
                    temp_f = current.get('temp_F', 'N/A')
                    desc = current.get('weatherDesc', [{}])[0].get('value', 'N/A')
                    humidity = current.get('humidity', 'N/A')
                    wind_mph = current.get('windspeedMiles', 'N/A')
                    
                    result = f"""Title:Current Weather in {location.title()}
Snippet:Temperature: {temp_f}°F ({temp_c}°C)
Weather: {desc}
Humidity: {humidity}%
Wind: {wind_mph} mph
Link:https://wttr.in/{quote(location)}"""
                    return result
        except Exception as e:
            print(f"Weather API error: {e}")
        
        return None
    
    def _web_search(self, query: str) -> str:
        """Fallback web search using DuckDuckGo HTML"""
        try:
            # DuckDuckGo HTML search (more reliable than API)
            url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                results = []
                
                # Extract search results
                for result in soup.find_all('div', class_='result__body')[:5]:
                    title_elem = result.find('a', class_='result__a')
                    snippet_elem = result.find('a', class_='result__snippet')
                    
                    if title_elem:
                        title = title_elem.text.strip()
                        link = title_elem.get('href', '')
                        snippet = snippet_elem.text.strip() if snippet_elem else "No description"
                        
                        results.append(f"Title:{title}\nSnippet:{snippet}\nLink:{link}")
                
                if results:
                    return "\n\n".join(results)
        except Exception as e:
            print(f"Web search error: {e}")
        
        # Final fallback
        return "No search results found. Please try a different query."

# Test the implementation
if __name__ == "__main__":
    searcher = WeatherSearch()
    result = searcher.search_weather("what's the weather in garden grove, ca?")
    print(result)