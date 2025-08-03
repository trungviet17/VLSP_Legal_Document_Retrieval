import json
from typing import Dict, List, Optional, Union, Any
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
import os

from src.utils.schema import Message, ROLE_VALUES, ToolDefinition
from src.llm.base import BaseLLM

load_dotenv()

class OpenAILLM(BaseLLM):
    _instances: Dict[str, "OpenAILLM"] = {}

    def __new__(cls, llm_config: Union[Dict, Any]):
        # Use a unique identifier based on the llm_config
        config_id = str(id(llm_config))
        if config_id not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(llm_config)
            cls._instances[config_id] = instance
        return cls._instances[config_id]

    def __init__(self, llm_config: Union[Dict, Any]):
        # Handle dictionary input
        if isinstance(llm_config, dict):
            # Convert dictionary to an object with attributes
            class ConfigObject:
                def __init__(self, config_dict):
                    for key, value in config_dict.items():
                        setattr(self, key, value)
            
            llm_config = ConfigObject(llm_config)
        
        self.model_name = getattr(llm_config, "model_name", "gpt-4.1-nano-2025-04-14")
        self.api_base = getattr(llm_config, "api_base", "https://api.openai.com/v1")
        self.api_key = getattr(llm_config, "api_key", self._get_api_key_from_env())
        self.max_tokens = getattr(llm_config, "max_tokens", 512)
        self.temperature = getattr(llm_config, "temperature", 0.1)
        
        if not self.api_key:
            raise ValueError(
                "API key is required for LLM. Provide the OpenAI API Key environment variable."
            )
            
        self.client = None

    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variable."""
        return os.getenv("OPENAI_API_KEY")

    def get_client(self):
        """Get or create OpenAI client"""
        if not self.client:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
            )
        return self.client

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),
    )
    async def chat_completion(
        self, 
        messages: List[Union[dict, Message]], 
        **kwargs
    ):
        """
        Generate a standard chat completion using LiteLLM without tools.
        
        Args:
            messages: List of messages for the conversation
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            The completion response from LiteLLM
        """
        client = self.get_client()
        formatted_messages = self.format_messages(messages)
        # Set default parameters
        params = {
            "model": kwargs.get("model", self.model_name),
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
            
        # Generate completion
        response = client.chat.completions.create(**params)
        return response

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),
    )
    async def chat_completion_with_tools(
        self, 
        messages: List[Union[dict, Message]], 
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: str = "auto",
        **kwargs
    ):
        """
        Generate a tool call message using LiteLLM with tools.
        
        Args:
            messages: List of messages for the conversation
            tools: Optional list of tools to make available
            tool_choice: How to choose tools ("auto", "required", or None)
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            The completion response from LiteLLM
        """
        client = self.get_client()
        formatted_messages = self.format_messages(messages)
        
        # Set default parameters
        params = {
            "model": kwargs.get("model", self.model_name),
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        if tools:
            params["tools"] = [tool.model_dump() for tool in tools]
            params["tool_choice"] = tool_choice
            
        # Generate completion
        response = client.chat.completions.create(**params)
        return response
        
    async def execute_tool_calls(self, response, tool_functions: Dict[str, callable]):
        """
        Execute tool calls from a response and create tool messages.
        
        Args:
            response: The completion response containing tool calls
            tool_functions: Dictionary mapping function names to callable functions
            
        Returns:
            List of tool messages with results
        """
        tool_messages = []
        
        if not hasattr(response.choices[0].message, 'tool_calls') or not response.choices[0].message.tool_calls:
            return tool_messages
            
        for tool_call in response.choices[0].message.tool_calls:
            function_call = tool_call.function
            function_name = function_call.name
            
            if function_name in tool_functions:
                try:
                    # Parse arguments and call the function
                    arguments = json.loads(function_call.arguments)
                    result = tool_functions[function_name](**arguments)
                    
                    # Create a tool message with the result
                    tool_message = Message.tool_message(
                        content=str(result),
                        name=function_name,
                        tool_call_id=tool_call.id
                    )
                    tool_messages.append(tool_message)
                except Exception as e:
                    # Handle errors in tool execution
                    error_message = f"Error executing {function_name}: {str(e)}"
                    tool_message = Message.tool_message(
                        content=error_message,
                        name=function_name,
                        tool_call_id=tool_call.id
                    )
                    tool_messages.append(tool_message)
                    
        return tool_messages
