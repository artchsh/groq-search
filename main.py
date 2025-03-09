import os
import json
import re
from groq import Groq
from dotenv import load_dotenv
from tools.tool_manager import (
    get_all_tool_definitions,
    get_tool_definitions,
    get_tool_function,
    detect_tool_call
)
from utils.logger import (
    log_user_input,
    log_system_message,
    log_model_request,
    log_model_response,
    log_tool_usage,
    log_error,
    log_assistant_response,
    log_conversation_state,
    log_debug
)

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
# Model definitions
MODEL = "llama-3.3-70b-versatile"  # Default model to use
ROUTING_MODEL = "llama-3.1-8b-instant"  # Model for routing decisions
GENERAL_MODEL = "llama-3.3-70b-versatile"  # Model for general questions

def route_query(query):
    """
    Routing logic to decide which tools might be needed to answer the query.
    
    Args:
        query (str): The user query
    
    Returns:
        list: List of tool names that should be used, or empty list for no tools
    """
    routing_prompt = f"""
    Given the following user query, determine if any tools are needed to answer it.
    
    Available tools:
    1. CALCULATE: For mathematical calculations and arithmetic
    2. WEB_SEARCH: For current information, facts, news, or data that requires searching the web
    
    For each query, respond with one of:
    - "TOOL: CALCULATE" if a calculation tool is needed
    - "TOOL: WEB_SEARCH" if a web search tool is needed for current information
    - "TOOL: WEB_SEARCH, CALCULATE" if both tools might be needed
    - "NO TOOL" if no tools are needed and you can answer from your knowledge
    
    User query: "{query}"
    
    Response (ONLY respond with one of the allowed formats above):
    """
    
    log_debug(f"Routing query: {query}")
    
    try:
        messages = [
            {"role": "system", "content": "You are a routing assistant that decides which tools are needed to answer user queries."},
            {"role": "user", "content": routing_prompt}
        ]
        
        log_model_request(ROUTING_MODEL, messages)
        
        response = client.chat.completions.create(
            model=ROUTING_MODEL,
            messages=messages,
            temperature=0.1,  # Low temperature for more consistent results
            max_completion_tokens=20  # We only need a short response
        )
        
        log_model_response(ROUTING_MODEL, response)
        
        routing_decision = response.choices[0].message.content.strip().upper()
        log_debug(f"Raw routing decision: {routing_decision}")
        
        # Check for tools in the routing decision
        tools = []
        if "WEB_SEARCH" in routing_decision:
            tools.append("web_search")
        if "CALCULATE" in routing_decision:
            tools.append("calculate")
        
        log_system_message(f"Routing decision: {', '.join(tools) if tools else 'no_tool'}")
        return tools
    except Exception as e:
        log_error("Error in routing query", e)
        return []

def run_conversation(user_prompt, conversation_history=None, show_feedback=True):
    """
    Run a conversation with the Groq AI model, allowing web search and calculation capabilities.
    
    Args:
        user_prompt: The user's input message
        conversation_history: Previous conversation messages
        show_feedback: Whether to return feedback messages about tool usage
    
    Returns:
        tuple: (response text, updated conversation history, feedback messages)
    """
    log_user_input(user_prompt)
    feedback_messages = []
    
    if conversation_history is None:
        conversation_history = [
            {
                "role": "system",
                "content": """You are a helpful AI assistant with web search and calculation capabilities.

IMPORTANT: When you need current information or facts, use the web_search function by calling it properly, NOT by writing it as text.
DO NOT write things like '<function=web_search>' or 'I'll use web_search' in your response.
Instead, use the provided function calling mechanism to actually invoke the tool.

Similarly, for calculations, use the calculate function properly through function calling.

Remember:
- For current information: Use web_search function
- For calculations: Use calculate function
- For questions you can answer from your knowledge: Just respond directly"""
            }
        ]
    
    # Add user message to conversation history
    conversation_history.append({
        "role": "user",
        "content": user_prompt,
    })
    
    log_conversation_state(conversation_history)
    
    # Determine if we need specific tools for this query
    suggested_tools = route_query(user_prompt)
    
    if show_feedback:
        if suggested_tools:
            feedback_messages.append(f"Routing decision: {', '.join(suggested_tools)}")
        else:
            feedback_messages.append("Routing decision: no_tool")
    
    # Always include both tools to give the model flexibility
    tools = get_all_tool_definitions()
    
    # Set the correct tool choice based on routing
    tool_choice = None
    if suggested_tools:
        if len(suggested_tools) == 1:
            # Direct the model to use the specific tool
            tool_choice = {
                "type": "function",
                "function": {"name": suggested_tools[0]}
            }
            if show_feedback:
                feedback_messages.append(f"Directing model to use {suggested_tools[0]}")
                log_system_message(f"Directing model to use {suggested_tools[0]}")
        else:
            # Multiple tools suggested, let model choose
            tool_choice = "auto" 
    else:
        # No tools needed but keep them available
        tool_choice = "auto"
    
    # If no tools suggested, still provide tools but with "none" preference
    if not suggested_tools:
        selected_model = GENERAL_MODEL
    else:
        selected_model = MODEL
    
    log_debug(f"Selected model: {selected_model}")
    
    # Make the initial API call to Groq
    try:
        log_model_request(selected_model, conversation_history, tools, tool_choice)
        
        response = client.chat.completions.create(
            model=selected_model,
            messages=conversation_history,
            stream=False,
            tools=tools,
            tool_choice=tool_choice,
            temperature=0.7,  # Slightly reduced temperature
            max_completion_tokens=4096
        )
        
        log_model_response(selected_model, response)
    except Exception as e:
        if show_feedback:
            feedback_messages.append(f"Error in API call: {str(e)}")
        log_error("Error in API call", e)
        return f"I encountered an error processing your request. Please try again.", conversation_history, feedback_messages
    
    # Extract the response and any tool call responses
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    
    if tool_calls:
        # Add the AI's response to the conversation
        conversation_history.append(response_message)
        
        # Process each tool call
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            tool_function = get_tool_function(function_name)
            
            if not tool_function:
                if show_feedback:
                    feedback_messages.append(f"Unknown tool called: {function_name}")
                log_error(f"Unknown tool called: {function_name}")
                continue
                
            function_args = json.loads(tool_call.function.arguments)
            
            # Call the appropriate tool and get the response
            if function_name == "web_search":
                query = function_args.get("query")
                if show_feedback:
                    feedback_messages.append(f"Assistant is using WebSearch, query is \"{query}\"")
                    log_system_message(f"Assistant is using WebSearch, query is \"{query}\"")
                
                try:
                    function_response = tool_function(query=query)
                    log_tool_usage("web_search", {"query": query}, function_response)
                except Exception as e:
                    function_response = json.dumps({"error": f"Search failed: {str(e)}"})
                    log_error(f"Web search failed for query: {query}", e)
            
            elif function_name == "calculate":
                expression = function_args.get("expression")
                if show_feedback:
                    feedback_messages.append(f"Assistant is using Calculator, expression is \"{expression}\"")
                    log_system_message(f"Assistant is using Calculator, expression is \"{expression}\"")
                
                try:
                    function_response = tool_function(expression)
                    log_tool_usage("calculate", {"expression": expression}, function_response)
                except Exception as e:
                    function_response = json.dumps({"error": f"Calculation failed: {str(e)}"})
                    log_error(f"Calculation failed for expression: {expression}", e)
            else:
                function_response = json.dumps({"error": f"Unknown function: {function_name}"})
                if show_feedback:
                    feedback_messages.append(f"Unknown function called: {function_name}")
                    log_error(f"Unknown function called: {function_name}")
            
            # Add the tool response to the conversation
            conversation_history.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )
            
        # Make a second API call with the updated conversation
        try:
            log_debug("Making second API call with tool results")
            log_model_request(MODEL, conversation_history)
            
            second_response = client.chat.completions.create(
                model=MODEL,
                messages=conversation_history
            )
            
            log_model_response(MODEL, second_response)
            
            # Add the final response to conversation history
            final_response = second_response.choices[0].message.content
            conversation_history.append({
                "role": "assistant",
                "content": final_response
            })
            
            log_assistant_response(final_response)
            return final_response, conversation_history, feedback_messages
        except Exception as e:
            if show_feedback:
                feedback_messages.append(f"Error in second API call: {str(e)}")
            log_error("Error in second API call", e)
            return "I encountered an error processing the search results. Please try again.", conversation_history, feedback_messages
    else:
        # If no tool was used through official API, check for text-based invocations
        content = response_message.content
        
        # Use the new tool detection function
        tool_call_info = detect_tool_call(content)
        
        if tool_call_info["tool_name"]:
            tool_name = tool_call_info["tool_name"]
            arguments = tool_call_info["arguments"]
            original_text = tool_call_info["original_text"]
            
            log_debug(f"Detected text-based tool call: {tool_name}")
            log_debug(f"Original text: {original_text}")
            log_debug(f"Extracted arguments: {arguments}")
            
            # Get the tool function
            tool_function = get_tool_function(tool_name)
            
            if tool_name == "web_search":
                query = arguments.get("query", "")
                if show_feedback:
                    feedback_messages.append(f"Assistant is using WebSearch via text pattern, query is \"{query}\"")
                    feedback_messages.append("Converting text pattern to proper tool call")
                    log_system_message(f"Assistant is using WebSearch via text pattern, query is \"{query}\"")
                    
                # Call the web search function manually
                try:
                    search_results = tool_function(query=query)
                    log_tool_usage("web_search", {"query": query}, search_results)
                except Exception as e:
                    search_results = json.dumps({"error": f"Search failed: {str(e)}"})
                    log_error(f"Web search failed for query: {query}", e)
                
                # Instead of adding the text pattern to history, create a proper tool call message
                tool_call_message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "manual_web_search_call",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": json.dumps({"query": query})
                            }
                        }
                    ]
                }
                conversation_history.append(tool_call_message)
                
                conversation_history.append({
                    "tool_call_id": "manual_web_search_call",
                    "role": "tool",
                    "name": "web_search",
                    "content": search_results,
                })
            
            elif tool_name == "calculate":
                expression = arguments.get("expression", "")
                if show_feedback:
                    feedback_messages.append(f"Assistant is using Calculator via text pattern, expression is \"{expression}\"")
                    feedback_messages.append("Converting text pattern to proper tool call")
                    log_system_message(f"Assistant is using Calculator via text pattern, expression is \"{expression}\"")
                    
                # Call the calculate function manually
                try:
                    calc_results = tool_function(expression)
                    log_tool_usage("calculate", {"expression": expression}, calc_results)
                except Exception as e:
                    calc_results = json.dumps({"error": f"Calculation failed: {str(e)}"})
                    log_error(f"Calculation failed for expression: {expression}", e)
                
                # Create a proper tool call message instead of text
                tool_call_message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "manual_calculate_call",
                            "type": "function",
                            "function": {
                                "name": "calculate",
                                "arguments": json.dumps({"expression": expression})
                            }
                        }
                    ]
                }
                conversation_history.append(tool_call_message)
                
                conversation_history.append({
                    "tool_call_id": "manual_calculate_call",
                    "role": "tool",
                    "name": "calculate",
                    "content": calc_results,
                })
            
            # Make a second API call with the updated conversation
            try:
                log_debug("Making second API call after text pattern detection")
                log_model_request(MODEL, conversation_history)
                
                second_response = client.chat.completions.create(
                    model=MODEL,
                    messages=conversation_history
                )
                
                log_model_response(MODEL, second_response)
                
                final_response = second_response.choices[0].message.content
                conversation_history.append({
                    "role": "assistant",
                    "content": final_response
                })
                
                log_assistant_response(final_response)
                return final_response, conversation_history, feedback_messages
            except Exception as e:
                if show_feedback:
                    feedback_messages.append(f"Error in second API call after text tool detection: {str(e)}")
                log_error("Error in second API call after text tool detection", e)
                return "I encountered an error processing the results. Please try again.", conversation_history, feedback_messages
        
        # If tools were suggested but not used (and no text pattern found)
        if suggested_tools:
            if show_feedback:
                feedback_messages.append("Detected suggested tools weren't used. Forcing tool usage...")
                log_system_message("Detected suggested tools weren't used. Forcing tool usage...")
            
            # Force the usage of the suggested tool
            primary_tool = suggested_tools[0]
            
            if primary_tool == "web_search":
                # Create a search query based on the user prompt
                search_query = user_prompt
                
                # Call web search with the user's query
                try:
                    search_results = get_tool_function("web_search")(query=search_query)
                    log_tool_usage("web_search", {"query": search_query}, search_results)
                except Exception as e:
                    search_results = json.dumps({"error": f"Search failed: {str(e)}"})
                    log_error(f"Forced web search failed for query: {search_query}", e)
                
                if show_feedback:
                    feedback_messages.append(f"Forcing WebSearch, query is \"{search_query}\"")
                    log_system_message(f"Forcing WebSearch, query is \"{search_query}\"")
                
                # Create a proper tool call message
                tool_call_message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "forced_web_search_call",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": json.dumps({"query": search_query})
                            }
                        }
                    ]
                }
                conversation_history.append(tool_call_message)
                
                conversation_history.append({
                    "tool_call_id": "forced_web_search_call",
                    "role": "tool",
                    "name": "web_search",
                    "content": search_results,
                })
                
            elif primary_tool == "calculate":
                # For calculation, we can't easily force it without a valid expression
                # Just note the failure
                if show_feedback:
                    feedback_messages.append("Could not force calculator usage without a valid expression")
                    log_system_message("Could not force calculator usage without a valid expression")
            
            # If we forced a tool, make a second API call
            if primary_tool == "web_search":
                try:
                    log_debug("Making second API call after forced tool usage")
                    log_model_request(MODEL, conversation_history)
                    
                    second_response = client.chat.completions.create(
                        model=MODEL,
                        messages=conversation_history
                    )
                    
                    log_model_response(MODEL, second_response)
                    
                    final_response = second_response.choices[0].message.content
                    conversation_history.append({
                        "role": "assistant",
                        "content": final_response
                    })
                    
                    log_assistant_response(final_response)
                    return final_response, conversation_history, feedback_messages
                except Exception as e:
                    if show_feedback:
                        feedback_messages.append(f"Error in second API call after forced tool usage: {str(e)}")
                    log_error("Error in second API call after forced tool usage", e)
                    return "I encountered an error processing the results. Please try again.", conversation_history, feedback_messages
            
        # If all else fails, just return the response as is
        if show_feedback and suggested_tools:
            feedback_messages.append("No tools were used despite routing suggestion")
            log_system_message("No tools were used despite routing suggestion")
            
        conversation_history.append({
            "role": "assistant",
            "content": content
        })
        
        log_assistant_response(content)
        return content, conversation_history, feedback_messages

def main():
    """
    Run the CLI interface for the chat assistant.
    """
    print("Welcome to Groq Assistant CLI! Type 'exit' to quit.")
    print("Assistant: Hello! How can I assist you today?")
    
    conversation_history = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant with web search and calculation capabilities. "
                      "When the user asks for information that requires current data or facts, "
                      "use the web_search function. For mathematical calculations, use the calculate function."
        }
    ]
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Assistant: Goodbye!")
            break
            
        response, conversation_history, feedback = run_conversation(user_input, conversation_history)
        
        # Display any feedback about tool usage
        for message in feedback:
            print(f"[System] {message}")
            
        print(f"Assistant: {response}")

if __name__ == "__main__":
    main()
