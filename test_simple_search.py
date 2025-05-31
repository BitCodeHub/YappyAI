#!/usr/bin/env python3
"""Test the simple web search"""

import asyncio
from app_v3_simple import SimpleWebSearch

async def test_search():
    queries = [
        "who just won the western conference finals in the nba?",
        "what is the weather in New York?",
        "latest AI news",
        "current bitcoin price"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        result = await SimpleWebSearch.search(query)
        print(result)

if __name__ == "__main__":
    asyncio.run(test_search())