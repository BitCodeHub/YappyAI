# 🐕 Yappy AI v3 - Real-time Web Search Deployment

## What's New in v3

✅ **AGGRESSIVE Web Search** - Always searches for current information
✅ **Multiple Search Sources**:
- DuckDuckGo instant answers
- Live weather data (wttr.in)
- Real-time cryptocurrency prices (Coinbase)
- News search with date filtering
- General web search fallback

✅ **Smart Search Detection** - Automatically searches for:
- Any question (ends with ?)
- Temporal queries (today, current, latest, 2024, 2025)
- Information queries (who is, what is, where is)
- Named entities (people, places, companies)
- Weather, news, prices, and more

✅ **Search Results Display** - Shows what was searched and found

## Quick Deploy to Render

1. **Update your repository** with the new files:
```bash
git add app_complete_v3.py static/yappy_v3.html DEPLOY_V3.md
git commit -m "feat: add v3 with aggressive real-time web search"
git push origin main
```

2. **Update Render deployment**:
   - Go to your Render dashboard
   - Update start command to: `python app_complete_v3.py`
   - The app will auto-deploy

3. **Access the new version**:
   - Visit: `https://your-app.onrender.com/static/yappy_v3.html`

## Local Testing

```bash
# Install dependencies (if not already installed)
pip install -r requirements_v2.txt

# Run v3
python app_complete_v3.py

# Open browser to http://localhost:8000
```

## Key Features

### 1. Always Current Information
- Searches web by default (toggle available)
- Shows search indicator when searching
- Displays top 3 search results

### 2. Enhanced Search Sources
- **Weather**: "What's the weather in Paris?"
- **Crypto**: "What's the Bitcoin price?"
- **News**: "Latest AI news today"
- **General**: "Who is the CEO of OpenAI?"

### 3. Search Results Integration
- LLM receives search results as context
- Instructs LLM to use current information
- Clear citations from search results

## Environment Variables

No additional environment variables needed! Just:
- `DATABASE_URL` (auto-set by Render)
- `PORT` (auto-set by Render)

## Testing Queries

Try these to see real-time search in action:

1. "What's the weather in New York?"
2. "What's the current Bitcoin price?"
3. "What happened in AI today?"
4. "Who won the latest NBA game?"
5. "What's the population of Tokyo in 2024?"
6. "Latest news about ChatGPT"
7. "Current stock price of Apple"
8. "What movies are playing now?"

## API Changes

The `/api/chat` endpoint now includes:
- `force_search` parameter (default: true)
- `web_searched` in response
- `search_results` array in response

## Troubleshooting

**If search isn't working:**
1. Check network connectivity
2. Look for search errors in console
3. Verify the "Always search web" toggle is ON
4. Try a simple query like "weather"

**If responses seem outdated:**
1. Make sure you're using v3 endpoints
2. Check that search results are being displayed
3. Verify API keys are set for your LLM provider

## Coming Soon

- [ ] More search providers
- [ ] Search result caching
- [ ] Search history
- [ ] Custom search filters
- [ ] Search API keys for premium sources

---

🐕 **Woof!** Yappy v3 is tail-waggingly excited to fetch current information for you!