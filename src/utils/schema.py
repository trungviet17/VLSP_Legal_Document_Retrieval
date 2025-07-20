from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

class Role(str, Enum):
    """Message role options"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

ROLE_VALUES = tuple(role.value for role in Role)
ROLE_TYPE = Literal[ROLE_VALUES]

class ToolDefinition(BaseModel):
    """Tool definition for function calling"""
    type: str = "function"
    function: Dict[str, Any]
    
    @classmethod
    def create_function_tool(
        cls, 
        name: str, 
        description: str, 
        properties: Dict[str, Dict[str, Any]], 
        required: Optional[List[str]] = None
    ) -> "ToolDefinition":
        """Create a function tool with the given parameters"""
        return cls(
            type="function",
            function={
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required or []
                }
            }
        )

class Message(BaseModel):
    """Represents a chat message in the conversation"""

    role: ROLE_TYPE = Field(...)
    content: Optional[str] = Field(default=None)
    tool_calls: Optional[List[Dict[str, Any]]] = Field(default=None)
    name: Optional[str] = Field(default=None)
    tool_call_id: Optional[str] = Field(default=None)

    def __add__(self, other) -> List["Message"]:
        """Supports operations like Message + list or Message + Message"""
        if isinstance(other, list):
            return [self] + other
        elif isinstance(other, Message):
            return [self, other]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __radd__(self, other) -> List["Message"]:
        """Supports operations like list + Message"""
        if isinstance(other, list):
            return other + [self]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(other).__name__}' and '{type(self).__name__}'"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Message to a dictionary format for API requests"""
        message = {"role": self.role}
        
        if self.content is not None:
            message["content"] = self.content
            
        if self.name is not None:
            message["name"] = self.name
            
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
            
        if self.tool_calls is not None:
            # Check if tool_calls are already dictionaries or need conversion
            message["tool_calls"] = [
                tool_call if isinstance(tool_call, dict) else tool_call.model_dump()
                for tool_call in self.tool_calls
            ]
            
        return message

    @classmethod
    def user_message(cls, content: str) -> "Message":
        """Create a user message"""
        return cls(role=Role.USER, content=content)

    @classmethod
    def system_message(cls, content: str) -> "Message":
        """Create a system message"""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def assistant_message(cls, content: Optional[str] = None, tool_calls: Optional[List[ToolDefinition]] = None) -> "Message":
        """Create an assistant message, optionally with tool calls"""
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def tool_message(cls, content: str, name: str, tool_call_id: str) -> "Message":
        """Create a tool message"""
        return cls(
            role=Role.TOOL,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
        )

    @classmethod
    def from_tool_calls(
        cls,
        tool_calls: List[Any],
        content: Union[str, List[str]] = "",
        **kwargs,
    ):
        """Create ToolCallsMessage from raw tool calls.

        Args:
            tool_calls: Raw tool calls from LLM
            content: Optional message content
        """
        formatted_calls = [
            {"id": call.id, "function": call.function.model_dump(), "type": "function"}
            for call in tool_calls
        ]
        return cls(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=formatted_calls,
            **kwargs,
        )

class Memory(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    max_messages: int = Field(default=100)

    def add_message(self, message: Message) -> None:
        """Add a message to memory"""
        self.messages.append(message)
        # Optional: Implement message limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def add_messages(self, messages: List[Message]) -> None:
        """Add multiple messages to memory"""
        self.messages.extend(messages)
        # Optional: Implement message limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def clear(self) -> None:
        """Clear all messages"""
        self.messages.clear()

    def get_recent_messages(self, n: int) -> List[Message]:
        """Get n most recent messages"""
        return self.messages[-n:]

    def to_dict_list(self) -> List[dict]:
        """Convert messages to list of dicts"""
        return [msg.to_dict() for msg in self.messages]