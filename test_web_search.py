#!/usr/bin/env python3
"""Test web search functionality"""

import asyncio
import sys
sys.path.append('.')

from app_complete_v2 import WebSearch

async def test_searches():
    """Test various search queries"""
    queries = [
        "what is the weather in New York",
        "latest AI news",
        "who is the president of France",
        "current bitcoin price",
        "what happened today in technology"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        results = await WebSearch.search(query)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']}")
            print(f"   {result['snippet']}")
            if result.get('url'):
                print(f"   URL: {result['url']}")

if __name__ == "__main__":
    asyncio.run(test_searches())