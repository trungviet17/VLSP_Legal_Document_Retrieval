from typing import Dict, List, Optional, Union, Any
import json
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type

from src.utils.configs import LLMSettings, CONFIGS
from src.utils.schema import Message, ROLE_VALUES, ToolDefinition
from openai import OpenAIError
from src.llm.base import BaseLLM


class VLLM(BaseLLM):
    _instances: Dict[str, "VLLM"] = {}

    def __new__(cls, llm_config: Union[LLMSettings, Dict]):
        # Use a unique identifier based on the llm_config
        config_id = str(id(llm_config))
        if config_id not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(llm_config)
            cls._instances[config_id] = instance
        return cls._instances[config_id]

    def __init__(self, llm_config: Union[LLMSettings, Dict]):
        # Handle dictionary input
        if isinstance(llm_config, dict):
            # Convert dictionary to an object with attributes
            class ConfigObject:
                def __init__(self, config_dict):
                    for key, value in config_dict.items():
                        setattr(self, key, value)
            
            llm_config = ConfigObject(llm_config)
        
        self.max_tokens = llm_config.max_tokens
        self.temperature = llm_config.temperature
        self.api_key = llm_config.api_key
        self.max_input_tokens = (
            llm_config.max_input_tokens
            if hasattr(llm_config, "max_input_tokens")
            else None
        )
        self.base_url = llm_config.base_url
        self.client = None
    
    def get_client(self):
        """Get or create OpenAI client"""
        from openai import OpenAI
        if not self.client:
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key or "dummy")
        return self.client

    # No need to redefine format_messages as it's inherited from BaseLLM

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),
    )
    def chat_completion(
        self, 
        messages: List[Union[dict, Message]], 
        **kwargs
    ):
        """
        Generate a standard chat completion using VLLM without tools.
        
        Args:
            messages: List of messages for the conversation
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            The completion response from VLLM
        """
        client = self.get_client()
        formatted_messages = self.format_messages(messages)
        
        # Set default parameters
        params = {
            "model": kwargs.get("model", client.models.list().data[0].id),
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
    def chat_completion_with_tools(
        self, 
        messages: List[Union[dict, Message]], 
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: str = "auto",
        **kwargs
    ):
        """
        Generate a chat completion using VLLM.
        
        Args:
            messages: List of messages for the conversation
            tools: Optional list of tools to make available
            tool_choice: How to choose tools ("auto", "required", or None)
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            The completion response from VLLM
        """
        client = self.get_client()
        formatted_messages = self.format_messages(messages)
        
        # Set default parameters
        params = {
            "model": kwargs.get("model", client.models.list().data[0].id),
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
        
    def execute_tool_calls(self, response, tool_functions: Dict[str, callable]):
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


