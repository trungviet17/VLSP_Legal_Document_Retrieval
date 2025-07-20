from typing import Dict, List, Optional, Union, Any
import json
from abc import ABC, abstractmethod
from tenacity import retry

from src.utils.schema import Message, ROLE_VALUES, ToolDefinition
from openai import OpenAIError


class BaseLLM(ABC):
    """
    Abstract base class for Language Model implementations.
    Provides common interface and utility methods for different LLM providers.
    """
    
    @abstractmethod
    def __init__(self, llm_config):
        """Initialize the LLM with configuration"""
        pass
    
    @abstractmethod
    def get_client(self):
        """Get or create client for the LLM provider"""
        pass
    
    @staticmethod
    def format_messages(messages: List[Union[dict, Message]]) -> List[dict]:
        """
        Format messages for LLM by converting them to OpenAI message format.

        Args:
            messages: List of messages that can be either dict or Message objects

        Returns:
            List[dict]: List of formatted messages in OpenAI format

        Raises:
            ValueError: If messages are invalid or missing required fields
            TypeError: If unsupported message types are provided
        """
        formatted_messages = []

        for message in messages:
            # Convert Message objects to dictionaries
            if isinstance(message, Message):
                message = message.to_dict()

            if isinstance(message, dict):
                # If message is a dict, ensure it has required fields
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")

                if "content" in message or "tool_calls" in message:
                    formatted_messages.append(message)
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # Validate all messages have required fields
        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"Invalid role: {msg['role']}")

        return formatted_messages
    
    @abstractmethod
    def chat_completion(self, messages: List[Union[dict, Message]], **kwargs):
        """Generate a standard chat completion"""
        pass
    
    @abstractmethod
    def chat_completion_with_tools(
        self, 
        messages: List[Union[dict, Message]], 
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: str = "auto",
        **kwargs
    ):
        """Generate a chat completion with tools"""
        pass
    
    @abstractmethod
    def execute_tool_calls(self, response, tool_functions: Dict[str, callable]):
        """Execute tool calls from a response"""
        pass