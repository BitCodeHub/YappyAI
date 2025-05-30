# 🌐 Intelligent Web Search Feature

Yappy now uses **AI-powered detection** to automatically determine when your question needs current/real-time web information - for ANY topic!

## 🧠 How It Works

**Smart AI Detection**: Instead of relying on keyword lists, Yappy uses AI to analyze your question and decide if it needs current information.

## ✨ What Automatically Triggers Web Search

### 🏖️ Travel & Planning
- "Plan a trip to Tokyo"
- "Best attractions in Paris" 
- "Hotels in New York"
- "Travel itinerary for Japan"

### 📰 Current Events & News
- "Latest news about..."
- "What's happening today"
- "Recent developments in..."
- "Breaking news"

### 🌡️ Real-time Information
- "Current weather in London"
- "Stock price of Apple"
- "Exchange rate USD to EUR"
- "Temperature forecast"

### 🛍️ Shopping & Prices
- "Best price for iPhone"
- "Where to buy..."
- "Product reviews for..."
- "Cost of..."

### 🎭 Events & Local Info
- "Events in San Francisco"
- "Restaurants near me"
- "Things to do in..."
- "Concert tickets"

## 🚀 How to Enable Full Web Search

For best results, start the SearxNG search service:

```bash
cd /Users/jimmylam/Downloads/agenticSeek-main
bash start_services.sh
```

This starts SearxNG on port 8080 for real-time web searches.

## 🔄 Fallback Behavior

If SearxNG isn't available, Yappy will:
- Provide enhanced AI responses for search-type queries
- Acknowledge limitations with current information
- Suggest where to find real-time data

## 📋 Example Queries

Try these examples:
- "Plan a 5-day trip to Italy"
- "Best restaurants in Tokyo"
- "Current weather in Los Angeles" 
- "Latest tech news"
- "Things to do in Amsterdam"

## 🔍 Response Format

Web search responses include:
- Comprehensive, structured information
- Bullet points and organized sections
- 🌐 **Information sourced from web search** indicator
- Specific details from search results

Enjoy exploring the web with Yappy! 🎉