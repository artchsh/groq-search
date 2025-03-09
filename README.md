# Groq Search Assistant

A CLI-based assistant that leverages Groq's LLM capabilities with tool augmentation for web search and calculations.

## Project Overview

This project implements an AI assistant using Groq's large language models, enhanced with the ability to perform web searches using Google's Custom Search API and execute mathematical calculations. The system features intelligent query routing, tool detection, and conversation management.

## Architecture

### Core Components

1. **Main Application (`main.py`)**: 
   - Contains the conversation loop and CLI interface
   - Implements the core `run_conversation` function for handling interactions
   - Includes routing logic to determine which tools to use

2. **Tool System (`tools/`)**: 
   - `tool_manager.py` - Central registry for tools with detection capabilities
   - `web_search.py` - Google Search API integration
   - `calculate.py` - Safe mathematical expression evaluation

3. **Utilities (`utils/`)**: 
   - `logger.py` - Comprehensive logging system
   - Various utility functions

### Models Used

- **Primary Model**: `llama-3.3-70b-versatile` - Used for general responses
- **Routing Model**: `llama-3.1-8b-instant` - Lightweight model for routing decisions

## Technical Features

### Query Routing System

The system uses a smaller, faster model to determine which tools might be required:
- Analyzes user queries to detect calculation or web search needs
- Routes requests to appropriate tool handlers
- Improves both response time and reduces unnecessary API calls

### Tool Execution Flow

1. Route query (determine if tools are needed)
2. Make initial API call to Groq with appropriate tools
3. If tool call is detected in the response, execute the tool
4. Make a second API call with the tool results
5. Return the final response to the user

### Tool Detection

Multiple approaches for detecting tool execution requests:
- Official function calling API through `tool_calls`
- Text pattern detection using regex for natural language tool requests
- Forced tool usage when routing suggests tools but model doesn't use them

### Logging System

Comprehensive logging with:
- Session-specific logs with timestamps
- Permanent log file
- Console output for key events
- Detailed recording of all API calls, tool usage, and conversation states

## Environment Requirements

- Python 3.8+
- Groq API key 
- Google Search API key and Custom Search Engine ID (for web search functionality)

## API Integrations

1. **Groq API**: 
   - Used for LLM capabilities
   - Supports function calling for tools

2. **Google Custom Search API**:
   - Powers the web search functionality
   - Returns structured search results

## Setup and Configuration

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with the following keys:
   ```
   GROQ_API_KEY=your_groq_api_key
   GOOGLE_SEARCH_API=your_google_search_api_key
   GOOGLE_CSE_ID=your_google_custom_search_engine_id
   ```
4. Run the application: `python main.py`

## Error Handling

The system implements comprehensive error handling:
- Tool execution errors are captured and returned as structured responses
- API call failures are logged and reported to the user
- Invalid tool usage is detected and remediated when possible

## Conversation Management

- Maintains conversation history for context
- Provides debugging feedback messages during tool execution
- Handles multiple tool calls in a single response

## Future Improvements

- Adding more tools (e.g., weather, translation)
- Web interface for easier interaction
- Streaming responses for better user experience
- Improved tool detection and execution
- More sophisticated routing logic
