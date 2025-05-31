"""SearxNG Search Tool - Copied from local app's searxSearch.py"""
import os
import json
import requests
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SearxSearch:
    def __init__(self, base_url: str = None):
        """Initialize SearxNG search with base URL"""
        self.base_url = base_url or os.getenv("SEARXNG_BASE_URL", "https://search.ononoki.org")
        self.timeout = 30
        
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search using SearxNG instance
        
        Args:
            query: Search query string
            num_results: Maximum number of results to return
            
        Returns:
            List of search results with title, snippet, and link
        """
        try:
            # Prepare search parameters
            params = {
                'q': query,
                'format': 'json',
                'language': 'en',
                'time_range': '',
                'safesearch': '0',
                'categories': 'general',
            }
            
            # Make request to SearxNG
            response = requests.get(
                f"{self.base_url}/search",
                params=params,
                timeout=self.timeout,
                headers={'User-Agent': 'AgenticSeek/1.0'}
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                # Extract results
                for result in data.get('results', [])[:num_results]:
                    results.append({
                        'title': result.get('title', ''),
                        'snippet': result.get('content', ''),
                        'link': result.get('url', ''),
                        'engine': result.get('engine', 'unknown')
                    })
                
                return results
            else:
                logger.error(f"SearxNG search failed with status {response.status_code}")
                return []
                
        except requests.exceptions.Timeout:
            logger.error("SearxNG search timed out")
            return []
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to SearxNG instance")
            return []
        except Exception as e:
            logger.error(f"SearxNG search error: {str(e)}")
            return []
    
    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for display"""
        if not results:
            return "No search results found."
        
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(f"{i}. **{result['title']}**")
            if result['snippet']:
                formatted.append(f"   {result['snippet']}")
            formatted.append(f"   Link: {result['link']}")
            formatted.append("")
        
        return "\n".join(formatted)

# Fallback web search using DuckDuckGo HTML parsing
def fallback_web_search(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """Fallback web search using DuckDuckGo HTML parsing"""
    try:
        import re
        from urllib.parse import unquote
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(
            f"https://html.duckduckgo.com/html/?q={query}",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            results = []
            
            # Parse results using regex
            result_pattern = r'<a class="result__a" href="([^"]+)"[^>]*>([^<]+)</a>.*?<a class="result__snippet"[^>]*>([^<]+)</a>'
            matches = re.findall(result_pattern, response.text, re.DOTALL)
            
            for match in matches[:num_results]:
                url = unquote(match[0])
                if url.startswith('//duckduckgo.com/l/?uddg='):
                    url = unquote(url.split('uddg=')[1].split('&')[0])
                
                results.append({
                    'title': match[1].strip(),
                    'snippet': match[2].strip(),
                    'link': url,
                    'engine': 'duckduckgo'
                })
            
            return results
    except Exception as e:
        logger.error(f"Fallback web search error: {str(e)}")
    
    return []