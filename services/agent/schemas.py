from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentRequest(Schema):
    session_id: str | None = Field(default=None, min_length=1, max_length=100)
    user_message: str = Field(min_length=1)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, session_id: str | None) -> str | None:
        if session_id is not None and not session_id.strip():
            raise ValueError("session_id must not be empty")
        return session_id


class ToolCallFunction(Schema):
    name: str = Field(min_length=1)
    # vLLM returns function arguments as a JSON-encoded string.
    arguments: str = "{}"


class ToolCall(Schema):
    id: str = Field(min_length=1)
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ChatMessage(Schema):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def validate_role_fields(self):
        if self.role == "tool":
            if not self.tool_call_id:
                raise ValueError("Tool messages require tool_call_id")
            if self.content is None:
                raise ValueError("Tool messages require content")
        elif self.tool_call_id is not None:
            raise ValueError("tool_call_id is only valid for tool messages")

        if self.role != "assistant" and self.tool_calls:
            raise ValueError("Only assistant messages can contain tool_calls")

        if self.role in ("system", "user") and self.content is None:
            raise ValueError(f"{self.role} messages require content")

        if self.role == "assistant" and self.content is None and not self.tool_calls:
            raise ValueError("Assistant messages require content or tool_calls")

        return self


class ToolResult(Schema):
    tool_call_id: str = Field(min_length=1)
    content: str
    is_error: bool = False


class SourceReference(Schema):
    source_type: Literal["unit", "document"]
    source_id: str = Field(min_length=1)
    score: float | None = None


class AgentResponse(Schema):
    session_id: str
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)