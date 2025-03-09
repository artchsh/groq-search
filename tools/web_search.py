import os
import json
import requests

def web_search(query):
    """
    Perform a web search using Google Search API.
    
    Args:
        query (str): The search query to look up
        
    Returns:
        str: JSON string with search results or error message
    """
    # Get API credentials from environment
    GOOGLE_API_KEY = os.environ.get("GOOGLE_SEARCH_API")
    GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID", "")
    
    if not GOOGLE_API_KEY:
        return "Error: Google Search API key is missing. Please add GOOGLE_SEARCH_API to your .env file."
    
    if not GOOGLE_CSE_ID:
        return "Error: Google Custom Search Engine ID is missing. Please add GOOGLE_CSE_ID to your .env file."
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'key': GOOGLE_API_KEY,
        'cx': GOOGLE_CSE_ID,
        'num': 5  # Number of results to return
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        search_results = response.json()
        
        formatted_results = []
        if 'items' in search_results:
            for item in search_results['items']:
                formatted_results.append({
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', '')
                })
            
            return json.dumps(formatted_results, indent=2)
        else:
            return "No results found"
    except Exception as e:
        return f"Search failed: {str(e)}"

def get_tool_definition():
    """
    Return the tool definition for use in the LLM API.
    """
    return {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information on a topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up",
                    }
                },
                "required": ["query"],
            },
        },
    }
