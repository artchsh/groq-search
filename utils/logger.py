"""
Logging utility for the assistant's activities.
"""
import os
import json
import logging
from datetime import datetime
import traceback
from typing import Any, Dict, List, Union

# Configure the logger
logger = logging.getLogger("llm_assistant")
logger.setLevel(logging.DEBUG)

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Create file handler with timestamp in filename
log_filename = os.path.join(logs_dir, f"llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

# Create another handler for the permanent log file
permanent_log = os.path.join(logs_dir, "llm.log")
permanent_handler = logging.FileHandler(permanent_log, encoding='utf-8')
permanent_handler.setLevel(logging.DEBUG)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter and set it to the handlers
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
file_handler.setFormatter(formatter)
permanent_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(permanent_handler)
logger.addHandler(console_handler)

def log_separator():
    """Add a separator line to the log for better readability"""
    logger.debug("-" * 80)

def prettify_json(obj):
    """Convert an object to a formatted JSON string for logging."""
    try:
        if isinstance(obj, str):
            # Try to parse the string as JSON first
            try:
                parsed = json.loads(obj)
                return json.dumps(parsed, indent=2, ensure_ascii=False)
            except:
                return obj
        else:
            return json.dumps(obj, indent=2, default=str, ensure_ascii=False)
    except:
        return str(obj)

def get_message_role(msg: Any) -> str:
    """
    Safely extract the role from a message object, handling various types.
    
    Args:
        msg: A message object which could be a dict, pydantic model, or other type
        
    Returns:
        str: The role, or "unknown" if not found
    """
    if isinstance(msg, dict):
        return msg.get("role", "unknown")
    else:
        # Handle pydantic models and other objects
        try:
            # Try attribute access
            return msg.role if hasattr(msg, "role") else "unknown"
        except:
            return "unknown"

def get_message_content(msg: Any) -> str:
    """
    Safely extract the content from a message object, handling various types.
    
    Args:
        msg: A message object which could be a dict, pydantic model, or other type
        
    Returns:
        str: The content, or "" if not found
    """
    if isinstance(msg, dict):
        return msg.get("content", "")
    else:
        # Handle pydantic models and other objects
        try:
            # Try attribute access for content
            if hasattr(msg, "content"):
                content = msg.content
                return content if content is not None else ""
            return ""
        except:
            return ""

def get_tool_calls(msg: Any) -> List:
    """
    Safely extract tool calls from a message object, handling various types.
    
    Args:
        msg: A message object which could be a dict, pydantic model, or other type
        
    Returns:
        list: The tool calls, or empty list if not found
    """
    if isinstance(msg, dict):
        return msg.get("tool_calls", [])
    else:
        # Handle pydantic models and other objects
        try:
            # Try attribute access
            return msg.tool_calls if hasattr(msg, "tool_calls") else []
        except:
            return []

def log_user_input(user_input):
    """Log user input"""
    logger.info(f"USER INPUT: {user_input}")
    log_separator()

def log_system_message(message):
    """Log a system message"""
    logger.info(f"SYSTEM: {message}")

def log_debug(message):
    """Log a debug message"""
    logger.debug(message)

def log_model_request(model_name, messages, tools=None, tool_choice=None):
    """Log the request being sent to the LLM"""
    logger.debug(f"REQUEST TO MODEL: {model_name}")
    logger.debug(f"MESSAGES:")
    
    try:
        for msg in messages:
            role = get_message_role(msg)
            content = get_message_content(msg)
            
            # Handle case where content is None but there are tool calls
            if content is None or content == "":
                tool_calls = get_tool_calls(msg)
                if tool_calls:
                    content = f"[TOOL CALLS: {len(tool_calls)}]"
            
            # Truncate very long content for readability
            if isinstance(content, str) and len(content) > 500:
                content = content[:500] + "... [truncated]"
                
            logger.debug(f"  [{role.upper()}]: {content}")
        
        if tools:
            logger.debug(f"TOOLS: {len(tools)} tools provided")
            for tool in tools:
                if isinstance(tool, dict) and "function" in tool:
                    logger.debug(f"  - {tool['function']['name']}: {tool['function']['description']}")
                elif hasattr(tool, "function") and hasattr(tool.function, "name"):
                    logger.debug(f"  - {tool.function.name}: {tool.function.description}")
        
        if tool_choice:
            logger.debug(f"TOOL_CHOICE: {prettify_json(tool_choice)}")
    except Exception as e:
        logger.error(f"Error logging model request: {str(e)}")
        logger.debug(traceback.format_exc())
    
    log_separator()

def log_model_response(model_name, response):
    """Log the response from the LLM"""
    logger.debug(f"RESPONSE FROM MODEL: {model_name}")
    
    try:
        # Extract relevant info from the response
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            
            if hasattr(choice, "message"):
                message = choice.message
                
                content = message.content if hasattr(message, "content") else None
                logger.debug(f"CONTENT: {content}")
                
                # Check for tool calls
                tool_calls = []
                if hasattr(message, "tool_calls") and message.tool_calls:
                    tool_calls = message.tool_calls
                
                if tool_calls:
                    logger.debug("TOOL CALLS:")
                    for tool_call in tool_calls:
                        # Handle different object types
                        if hasattr(tool_call, "function"):
                            function_name = tool_call.function.name if hasattr(tool_call.function, "name") else "unknown"
                            function_args = tool_call.function.arguments if hasattr(tool_call.function, "arguments") else "{}"
                        elif isinstance(tool_call, dict):
                            function_name = tool_call.get("function", {}).get("name", "unknown")
                            function_args = tool_call.get("function", {}).get("arguments", "{}")
                        else:
                            function_name = "unknown"
                            function_args = "{}"
                            
                        logger.debug(f"  FUNCTION: {function_name}")
                        try:
                            parsed_args = json.loads(function_args) if isinstance(function_args, str) else function_args
                            logger.debug(f"  ARGUMENTS: {prettify_json(parsed_args)}")
                        except:
                            logger.debug(f"  ARGUMENTS: {function_args}")
    except Exception as e:
        logger.error(f"Error logging model response: {str(e)}")
        logger.debug(traceback.format_exc())
    
    log_separator()

def log_tool_usage(tool_name, arguments, result):
    """Log tool usage with inputs and outputs"""
    logger.debug(f"TOOL EXECUTION: {tool_name}")
    logger.debug(f"ARGUMENTS: {prettify_json(arguments)}")
    logger.debug(f"RESULT: {prettify_json(result)}")
    log_separator()

def log_error(error_message, exception=None):
    """Log an error with optional exception details"""
    logger.error(f"ERROR: {error_message}")
    if exception:
        logger.error(f"EXCEPTION: {str(exception)}")
        logger.debug(traceback.format_exc())
    log_separator()

def log_assistant_response(response):
    """Log the assistant's final response"""
    logger.info(f"ASSISTANT RESPONSE: {response}")
    log_separator()

def log_conversation_state(conversation_history):
    """Log the current state of the conversation history"""
    logger.debug("CURRENT CONVERSATION STATE:")
    try:
        for i, msg in enumerate(conversation_history):
            role = get_message_role(msg)
            content = get_message_content(msg)
            
            # Handle case where content is None but there are tool calls
            if content is None or content == "":
                tool_calls = get_tool_calls(msg)
                if tool_calls:
                    content = f"[TOOL CALLS: {len(tool_calls)}]"
            
            # Truncate very long content for readability
            if isinstance(content, str) and len(content) > 500:
                content = content[:500] + "... [truncated]"
                
            logger.debug(f"  {i}. [{role.upper()}]: {content}")
    except Exception as e:
        logger.error(f"Error logging conversation state: {str(e)}")
        logger.debug(traceback.format_exc())
        
    log_separator()
