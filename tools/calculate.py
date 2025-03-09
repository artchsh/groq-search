import json
import re

def calculate(expression):
    """
    Tool to evaluate a mathematical expression safely.
    
    Args:
        expression (str): Mathematical expression to evaluate
        
    Returns:
        str: JSON string with calculation result or error message
    """
    try:
        # Safely evaluate mathematical expressions
        # First check if the expression contains only allowed characters
        if not re.match(r'^[\d\+\-\*\/\(\)\.\s\^\%]*$', expression):
            return json.dumps({"error": "Invalid characters in expression"})
        
        # Replace ^ with ** for exponentiation
        expression = expression.replace('^', '**')
        
        # Use eval with limited scope to prevent code execution
        result = eval(expression, {"__builtins__": {}})
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": f"Error calculating: {str(e)}"})

def get_tool_definition():
    """
    Return the tool definition for use in the LLM API.
    """
    return {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        },
    }
