#!/usr/bin/env python3
"""
Patch to add context awareness and better web search to app_db.py
Run this to update your existing app_db.py with v4 features
"""

import re

def add_context_awareness():
    """Add the context awareness code to app_db.py"""
    
    # Read the current app_db.py
    with open('app_db.py', 'r') as f:
        content = f.read()
    
    # Add helper function for context extraction after the chat endpoint definition
    context_code = '''
# Helper function to extract context from conversation
def extract_context_from_messages(messages, current_query):
    """Extract context from previous messages"""
    context = {
        "location": None,
        "topic": None,
        "previous_query": None
    }
    
    if not messages:
        return context
    
    # Look at last 3 messages
    recent = messages[-3:] if len(messages) > 3 else messages
    
    for msg in recent:
        user_msg = msg.get("user_message", "").lower()
        
        # Extract location from weather queries
        if "weather" in user_msg:
            import re
            loc_match = re.search(r'weather (?:in |at |for )?([\\w\\s,]+)', user_msg, re.IGNORECASE)
            if loc_match:
                context["location"] = loc_match.group(1).strip()
                context["topic"] = "weather"
        
        # Track NBA/sports queries
        if any(word in user_msg for word in ["nba", "basketball", "game", "score"]):
            context["topic"] = "nba"
    
    # Get previous query
    if messages:
        context["previous_query"] = messages[-1].get("user_message", "")
    
    return context

# Enhanced web search with context
async def search_with_context(query, context):
    """Perform context-aware web search"""
    import requests
    from urllib.parse import quote
    from datetime import datetime
    
    web_results = []
    
    # Handle follow-up queries
    if context.get("topic") == "weather" and context.get("location"):
        # Check for forecast queries
        if any(word in query.lower() for word in ["7 day", "forecast", "next", "week"]):
            location = context["location"]
            try:
                weather_url = f"https://wttr.in/{quote(location)}?format=j1"
                response = requests.get(weather_url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    forecast = data.get("weather", [])
                    
                    result = f"7-Day Forecast for {location}:\\n"
                    result += f"(Today is {datetime.now().strftime('%A, %B %d, %Y')})\\n\\n"
                    
                    for i, day in enumerate(forecast[:7]):
                        date = day.get("date", "")
                        max_temp = day.get("maxtempF", "N/A")
                        min_temp = day.get("mintempF", "N/A")
                        desc = day.get("hourly", [{}])[4].get("weatherDesc", [{}])[0].get("value", "N/A")
                        
                        try:
                            from datetime import datetime
                            date_obj = datetime.strptime(date, "%Y-%m-%d")
                            day_name = date_obj.strftime("%A, %B %d")
                        except:
                            day_name = f"Day {i+1}"
                        
                        result += f"{day_name}: High {max_temp}°F, Low {min_temp}°F - {desc}\\n"
                    
                    web_results.append(result)
                    return web_results
            except Exception as e:
                print(f"Forecast error: {e}")
    
    # NBA/Sports queries
    if "nba" in query.lower() or "basketball" in query.lower():
        try:
            nba_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
            response = requests.get(nba_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                events = data.get("events", [])
                
                result = f"NBA Games Today ({datetime.now().strftime('%B %d, %Y')}):\\n\\n"
                
                if not events:
                    result += "No games scheduled today.\\n"
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
                            
                            result += f"{team1} vs {team2}\\n"
                            if status == "Final":
                                result += f"Final: {score1} - {score2}\\n"
                                winner = team1 if int(score1) > int(score2) else team2
                                result += f"Winner: {winner}\\n\\n"
                            else:
                                result += f"Status: {status}\\n\\n"
                
                web_results.append(result)
        except Exception as e:
            print(f"NBA search error: {e}")
    
    # Bitcoin/Crypto queries
    if any(word in query.lower() for word in ["btc", "bitcoin", "crypto", "eth", "ethereum"]):
        try:
            crypto_url = "https://api.coinbase.com/v2/exchange-rates?currency=USD"
            response = requests.get(crypto_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                rates = data.get("data", {}).get("rates", {})
                
                btc_price = 1 / float(rates.get("BTC", 1))
                eth_price = 1 / float(rates.get("ETH", 1))
                
                result = f"Current Cryptocurrency Prices:\\n"
                result += f"Bitcoin (BTC): ${btc_price:,.2f} USD\\n"
                result += f"Ethereum (ETH): ${eth_price:,.2f} USD\\n"
                result += f"(Live prices from Coinbase)"
                
                web_results.append(result)
        except Exception as e:
            print(f"Crypto search error: {e}")
    
    # Regular web search if no context match
    if not web_results:
        # Use existing web search code
        pass
    
    return web_results
'''
    
    # Find where to insert the context code (after the chat endpoint)
    insert_pos = content.find('@app.post("/api/chat"')
    if insert_pos == -1:
        print("❌ Could not find chat endpoint")
        return False
    
    # Find the beginning of the function
    insert_pos = content.rfind('\n', 0, insert_pos)
    
    # Insert the context code
    new_content = content[:insert_pos] + "\n" + context_code + "\n" + content[insert_pos:]
    
    # Update the chat endpoint to use context
    # Find the chat function
    chat_start = new_content.find('async def chat(')
    chat_end = new_content.find('\n@app.', chat_start)
    if chat_end == -1:
        chat_end = new_content.find('\nif __name__', chat_start)
    
    chat_function = new_content[chat_start:chat_end]
    
    # Add context extraction after getting conversation messages
    context_check = '''
        # Extract context from conversation history
        context = extract_context_from_messages(conversation_messages, request.message)
        
        # Check if this is a follow-up query
        is_followup = False
        if context.get("previous_query") and any(word in request.message.lower() for word in ["next", "7 day", "forecast", "more", "what about"]):
            is_followup = True
            print(f"Follow-up detected. Previous topic: {context.get('topic')}, Location: {context.get('location')}")
        
        # Perform context-aware search
        if is_followup or needs_web_search or starts_with_question:
            try:
                context_results = await search_with_context(request.message, context)
                if context_results:
                    web_context = "\\n\\n🔍 Web Search Results:\\n" + "\\n".join(context_results) + "\\n\\n"
                    print(f"Context-aware search found {len(context_results)} results")
            except Exception as e:
                print(f"Context search error: {e}")
'''
    
    # Find where to insert context check (after getting conversation messages)
    insert_marker = "conversation_messages = conversation.messages or []"
    marker_pos = new_content.find(insert_marker)
    if marker_pos != -1:
        insert_at = new_content.find('\n', marker_pos) + 1
        new_content = new_content[:insert_at] + "\n" + context_check + "\n" + new_content[insert_at:]
    
    # Update system prompt to include current date
    date_prompt = '''
        # Update system prompt with current date
        from datetime import datetime
        current_date = datetime.now().strftime("%B %d, %Y")
        self.system_prompt = f"""You are Yappy, an incredibly friendly AI assistant. Today's date is {current_date}.
        
        When users ask follow-up questions (like "next 7 days" after weather), understand they're referring to the previous topic.
        Always use the current date for any date-related responses.
        When you receive web search results, use them as the primary source of information.
        
        Be helpful, accurate, and maintain your cheerful personality!"""
'''
    
    # Find LLMHandler init
    handler_pos = new_content.find('def __init__(self):')
    if handler_pos != -1:
        # Find the system_prompt line
        prompt_pos = new_content.find('self.system_prompt =', handler_pos)
        if prompt_pos != -1:
            # Replace the prompt initialization
            prompt_end = new_content.find('"""', prompt_pos + 20) + 3
            old_prompt = new_content[prompt_pos:prompt_end]
            new_content = new_content.replace(old_prompt, date_prompt.strip())
    
    # Write the updated file
    with open('app_db_v4_patched.py', 'w') as f:
        f.write(new_content)
    
    print("✅ Created app_db_v4_patched.py with context awareness!")
    print("📝 To use it:")
    print("1. Review the changes: diff app_db.py app_db_v4_patched.py")
    print("2. Backup original: cp app_db.py app_db_backup.py")
    print("3. Apply patch: cp app_db_v4_patched.py app_db.py")
    print("4. Commit and push: git add app_db.py && git commit -m 'fix: add context awareness' && git push")
    
    return True

if __name__ == "__main__":
    print("🔧 Patching app_db.py with v4 context awareness...")
    success = add_context_awareness()
    if not success:
        print("❌ Patch failed. Please apply changes manually.")