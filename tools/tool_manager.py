"""
Tool manager for loading and registering tools.
"""
from typing import Dict, List, Callable, Any
import importlib
import re
import json

# Dictionary to store tool functions, keyed by tool name
_tools: Dict[str, Callable] = {}

# Dictionary to store tool definitions for API calls
_tool_definitions: Dict[str, Dict[str, Any]] = {}

def register_tool(name: str, function: Callable, definition: Dict[str, Any]):
    """Register a tool function and its definition."""
    _tools[name] = function
    _tool_definitions[name] = definition

def get_all_tool_definitions() -> List[Dict[str, Any]]:
    """Get all registered tool definitions."""
    return list(_tool_definitions.values())

def get_tool_definitions(tool_names: List[str]) -> List[Dict[str, Any]]:
    """Get definitions for specified tools."""
    return [_tool_definitions[name] for name in tool_names if name in _tool_definitions]

def get_tool_function(name: str) -> Callable:
    """Get a tool function by name."""
    return _tools.get(name)

def load_tools():
    """Load all available tools from the tools package."""
    # Import and register web_search
    from tools import web_search
    register_tool(
        "web_search", 
        web_search.web_search, 
        web_search.get_tool_definition()
    )
    
    # Import and register calculate
    from tools import calculate
    register_tool(
        "calculate", 
        calculate.calculate, 
        calculate.get_tool_definition()
    )

def detect_tool_call(text: str) -> dict:
    """
    Detect if the text contains a tool call in various formats.
    
    Returns dict with:
        - tool_name: name of detected tool or None
        - arguments: dict of arguments or None
        - original_text: the matching text portion
    """
    # Different patterns for function call detection
    patterns = [
        # <function=name [{"arg":"value"}]>
        r'<function=(\w+)\s+\[(.*?)\]',
        # <function=name [{"arg":"value"}]>...</function>
        r'<function=(\w+)\s+\[(.*?)\](?:.*?)</function>',
        # <function=name {"arg":"value"}>
        r'<function=(\w+)\s+({.*?})>',
        # function name({"arg":"value"})
        r'function\s+(\w+)\(([^)]*)\)',
        # name({"arg":"value"})
        r'(\w+)\(([^)]*)\)',
        # <tool:name query="value">
        r'<tool:(\w+)[^>]*>([^<]*)</tool>',
        # Using name to query for "value"
        r'Using (\w+) to (?:search|query|calculate) (?:for )?"([^"]*)"',
        # I'll search for "query"
        r'I\'ll search (?:for|about) ["\']([^"\']+)["\']',
        # Let me search/calculate...
        r'Let me (search|calculate) ["\']([^"\']+)["\']',
        # Let me use web_search/calculate...
        r'Let me use (\w+)[^\n]+"([^"]+)"',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            # Special case for the "Let me search" pattern
            if pattern == r'I\'ll search (?:for|about) ["\']([^"\']+)["\']':
                query = match.group(1)
                return {
                    "tool_name": "web_search",
                    "arguments": {"query": query},
                    "original_text": match.group(0)
                }
            
            # Special case for "Let me calculate/search" pattern
            if pattern == r'Let me (search|calculate) ["\']([^"\']+)["\']':
                tool_type = match.group(1).lower()
                value = match.group(2)
                
                if tool_type == "search":
                    return {
                        "tool_name": "web_search",
                        "arguments": {"query": value},
                        "original_text": match.group(0)
                    }
                elif tool_type == "calculate":
                    return {
                        "tool_name": "calculate",
                        "arguments": {"expression": value},
                        "original_text": match.group(0)
                    }
            
            # The standard pattern with tool name and arguments
            if len(match.groups()) >= 2:
                tool_name = match.group(1).lower()
                args_text = match.group(2)
                
                # Convert to proper tool name if variation
                if tool_name in ["websearch", "search", "googlesearch"]:
                    tool_name = "web_search"
                elif tool_name in ["calc", "calculator"]:
                    tool_name = "calculate"
                
                # Only process if it's a known tool
                if tool_name in _tools:
                    # Try to parse arguments
                    if tool_name == "web_search":
                        # Extract query from different formats
                        query_match = re.search(r'"query":\s*"([^"]+)"', args_text)
                        if query_match:
                            query = query_match.group(1)
                        else:
                            # Try alternate patterns
                            query_match = re.search(r'"([^"]+)"', args_text)
                            if query_match:
                                query = query_match.group(1)
                            else:
                                # Just use the raw text if nothing else works
                                query = args_text.strip('"\'{}[] ')
                        
                        return {
                            "tool_name": "web_search",
                            "arguments": {"query": query},
                            "original_text": match.group(0)
                        }
                    
                    elif tool_name == "calculate":
                        # Extract expression from different formats
                        expr_match = re.search(r'"expression":\s*"([^"]+)"', args_text)
                        if expr_match:
                            expression = expr_match.group(1)
                        else:
                            expr_match = re.search(r'"([^"]+)"', args_text)
                            if expr_match:
                                expression = expr_match.group(1)
                            else:
                                # Clean up the expression
                                expression = args_text.strip('"\'{}[] ')
                        
                        return {
                            "tool_name": "calculate",
                            "arguments": {"expression": expression},
                            "original_text": match.group(0)
                        }
    
    # Look for mentions of web search without explicit query format
    search_indicators = [
        r'I need to search for (.*?)[\.\n]',
        r'I should search for (.*?)[\.\n]',
        r'I will search for (.*?)[\.\n]',
        r'Let me search for (.*?)[\.\n]',
        r'I would need to look up (.*?)[\.\n]'
    ]
    
    for pattern in search_indicators:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            query = match.group(1).strip().strip('"\'')
            return {
                "tool_name": "web_search",
                "arguments": {"query": query},
                "original_text": match.group(0)
            }
    
    return {"tool_name": None, "arguments": None, "original_text": None}

# Initialize tools when this module is imported
load_tools()
