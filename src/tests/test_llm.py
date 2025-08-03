import asyncio
from src.llm import OpenAILLM
from src.utils.schema import Message, ToolDefinition


def get_weather(location: str, unit: str):
    return f"Getting the weather for {location} in {unit}..."

def perform_search(query: str, max_results: int = 5):
    """DuckDuckGo search engine."""
    from duckduckgo_search import DDGS
    return DDGS().text(query, max_results=max_results)

# Map of tool functions
tool_functions = {
    "get_weather": get_weather,
    "perform_search": perform_search
}

# Define the weather tool
weather_tool = ToolDefinition(
    type="function",
    function={
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and state, e.g., 'San Francisco, CA'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location", "unit"]
        }
    }
)

# Define the search tool
search_tool = ToolDefinition(
    type="function",
    function={
        "name": "perform_search",
        "description": "Search the web for information using DuckDuckGo",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "max_results": {"type": "integer", "description": "Maximum number of results to return", "default": 5}
            },
            "required": ["query"]
        }
    }
)

async def main():
    # Initialize LiteLLM with custom config
    llm_config = {
        "max_tokens": 1024,
        "temperature": 0.7,
        # API key will be taken from environment variable LITELLM_API_KEY
    }
    
    # Create LiteLLM instance
    litellm = OpenAILLM(llm_config=llm_config)
    
    # Create messages
    messages = [
        Message.system_message("You are a helpful assistant that can provide weather information and search the web."),
        Message.user_message("What's the weather like in San Francisco? Also, find me information about Golden Gate Bridge.")
    ]
    
    # First test a simple completion without tools
    print("Testing basic chat completion...")
    basic_response = await litellm.chat_completion(
        messages=[Message.user_message("Tell me about artificial intelligence in 2-3 sentences.")]
    )
    print("\nBasic response:")
    print(basic_response.choices[0].message.content)
    
    # Generate completion with multiple tools
    print("\nGenerating completion with tool support...")
    response = await litellm.chat_completion_with_tools(
        messages=messages,
        tools=[weather_tool, search_tool], 
        tool_choice="auto"
    )
    
    print("\nResponse:")
    print(f"Content: {response.choices[0].message.content}")
    
    # Check if tool calls were made
    if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
        print("\nTool calls:")
        for tool_call in response.choices[0].message.tool_calls:
            print(f"Tool: {tool_call.function.name}")
            print(f"Arguments: {tool_call.function.arguments}")
        
        # Execute tool calls
        print("\nExecuting tool calls...")
        tool_messages = await litellm.execute_tool_calls(response, tool_functions)
        
        # Print tool messages
        print("\nTool messages:")
        for msg in tool_messages:
            print(f"{msg.name}: {msg.content}")
        
        # Continue the conversation with tool results
        follow_up_messages = messages + [
            Message(
                role="assistant",
                content=response.choices[0].message.content,
                tool_calls=[{
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in response.choices[0].message.tool_calls]
            )
        ] + tool_messages
        
        # Generate follow-up response
        print("\nGenerating follow-up response...")
        follow_up = await litellm.chat_completion(
            messages=follow_up_messages
        )
        
        print("\nFinal response:")
        print(follow_up.choices[0].message.content)
    else:
        print("No tool calls were made.")

if __name__ == "__main__":
    asyncio.run(main())