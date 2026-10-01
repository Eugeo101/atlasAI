import json
import logging
from uuid import uuid4
from llm_client import LLMClient # interact with vLLM
from prompts import SYSTEM_PROMPT, ROUTER_SYSTEM_PROMPT # Prompt
from schemas import AgentResponse, ChatMessage, SourceReference, ToolCall # Schemas
from tools import ToolRegistry # Tool
from db_memory import DatabaseConversationMemory


logger = logging.getLogger(__name__)

FINAL_MAX_TOKENS = 1500


class AgentOrchestrator:
    def __init__(
        self,
        llm_client: LLMClient,
        memory: DatabaseConversationMemory,
        registry: ToolRegistry,
        max_tool_rounds: int = 3,
    ) -> None:
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1")

        self._llm_client = llm_client
        self._memory = memory
        self._registry = registry
        self._max_tool_rounds = max_tool_rounds

    def run(
        self,
        user_query: str,
        session_id: str | None = None,
    ) -> AgentResponse:
        if session_id is None:
            session_id = str(uuid4())
        if not session_id.strip():
            raise ValueError("session_id must not be empty")
        if len(session_id) > 100:
            raise ValueError("session_id must not exceed 100 characters")
        if not user_query.strip():
            raise ValueError("user_query must not be empty")

        stored_history = self._memory.get_history(session_id)
        history = self._compact_history(stored_history)
        user_message = ChatMessage(role="user", content=user_query)
        sources: list[SourceReference] = []
        evidence: list[dict] = []
        tool_definitions = self._registry.openai_tool_definitions()
        definitions_by_name = {
            item["function"]["name"]: item for item in tool_definitions
        }
        pending_tools = self._required_tools(user_query=user_query)
        if len(pending_tools) > self._max_tool_rounds:
            raise RuntimeError(
                f"Request requires {len(pending_tools)} tool calls, exceeding "
                f"the configured limit of {self._max_tool_rounds}"
            )

        tool_request_context = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            *history,
            user_message,
        ]

        while pending_tools:
            tool_name = pending_tools.pop(0)
            response = self._llm_client.chat_completion(
                messages=self._serialize_messages(tool_request_context),
                tools=[definitions_by_name[tool_name]],
                tool_choice="required",
            )
            assistant_message = self._parse_assistant_message(response)

            if len(assistant_message.tool_calls) != 1:
                raise RuntimeError(
                    f"vLLM must return exactly one structured {tool_name} call; "
                    "the response may contain tool markup as plain text instead."
                )

            tool_call = assistant_message.tool_calls[0]
            if tool_call.function.name != tool_name:
                raise RuntimeError(
                    f"Expected {tool_name}, but vLLM requested "
                    f"{tool_call.function.name}"
                )

            result = self._execute_tool(tool_call)
            sources.extend(self._extract_sources(tool_name, result))
            evidence.append(self._compact_tool_result(tool_name, result))

        final_user_message = ChatMessage(
            role="user",
            content=(
                f"Question: {user_query}\n\n"
                "Retrieved evidence (JSON):\n"
                f"{json.dumps(evidence, ensure_ascii=False, separators=(',', ':'))}"
            ),
        )
        api_messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            *history,
            final_user_message,
        ]
        response = self._llm_client.chat_completion(
            messages=self._serialize_messages(api_messages),
            max_tokens=FINAL_MAX_TOKENS,
        )
        assistant_message = self._parse_assistant_message(response)
        if assistant_message.tool_calls:
            raise RuntimeError("vLLM returned tool calls when no tools were offered")
        if assistant_message.content is None:
            raise ValueError("vLLM returned an empty final answer")

        self._memory.remember_turn(
            session_id,
            [
                ChatMessage(role="user", content=user_query),
                ChatMessage(role="assistant", content=assistant_message.content),
            ],
        )
        return AgentResponse(
            session_id=session_id,
            answer=assistant_message.content,
            sources=self._unique_sources(sources),
        )

    @staticmethod
    def _compact_history(history: list[ChatMessage]) -> list[ChatMessage]:
        return [
            message
            for message in history
            if message.role in ("user", "assistant")
            and not message.tool_calls
            and message.content
        ]

    @staticmethod
    def _compact_tool_result(tool_name: str, result: dict) -> dict:
        if "error" in result:
            return {"tool": tool_name, "error": result["error"]}

        compact_results = []
        for item in result.get("results", []):
            if not isinstance(item, dict):
                continue

            payload = item.get("payload", item)
            text = item.get("text") or payload.get("text") or ""
            if tool_name == "units_search":
                fields = (
                    "unit_id",
                    "city",
                    "district",
                    "unit_type",
                    "price",
                    "bedrooms",
                    "bathrooms",
                    "area_sqm",
                    "is_available",
                )
            else:
                fields = ("filename", "doc_type", "location", "chunk_index")

            compact_results.append(
                {
                    **{
                        field: payload[field]
                        for field in fields
                        if field in payload
                    },
                    "rerank_score": item.get("rerank_score"),
                    "text": text,
                }
            )

        return {"tool": tool_name, "results": compact_results}

    def _required_tools(self, user_query: str) -> list[str]:
        response = self._llm_client.chat_completion(
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
            temperature=0.0,
            max_tokens=80,
        )

        content = response.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Tool router returned no JSON content")

        try:
            plan = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Tool router returned invalid JSON") from exc

        names = plan.get("required_tools") if isinstance(plan, dict) else None
        allowed = {"units_search", "documents_search"}

        if (
            not isinstance(names, list)
            or any(not isinstance(name, str) or name not in allowed for name in names)
            or len(names) != len(set(names))
        ):
            raise RuntimeError("Tool router returned an invalid required_tools list")

        return names

    @staticmethod
    def _serialize_messages(messages: list[ChatMessage]) -> list[dict]:
        serialized = []

        for message in messages:
            data = message.model_dump(mode="json", exclude_none=True)
            if not message.tool_calls:
                data.pop("tool_calls", None)
            serialized.append(data)

        return serialized

    @staticmethod
    def _parse_assistant_message(message: dict) -> ChatMessage:
        if not isinstance(message, dict):
            raise ValueError("vLLM returned an invalid assistant message")

        normalized = {
            "role": "assistant",
            "content": message.get("content"),
            "tool_calls": [],
        }

        for raw_call in message.get("tool_calls") or []:
            function = raw_call.get("function") or {}
            arguments = function.get("arguments", "{}")

            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)

            normalized["tool_calls"].append(
                {
                    "id": raw_call.get("id"),
                    "type": raw_call.get("type", "function"),
                    "function": {
                        "name": function.get("name"),
                        "arguments": arguments,
                    },
                }
            )

        return ChatMessage.model_validate(normalized)
    
    def _execute_tool(self, tool_call: ToolCall) -> dict:
        tool_name = tool_call.function.name

        try:
            return self._registry.execute(
                name=tool_name,
                arguments_json=tool_call.function.arguments,
            )
        except Exception:
            # Log the underlying failure locally, but don't expose internals
            # such as connection strings or host details to the model/user.
            logger.exception("Tool execution failed: %s", tool_name)
            return {
                "error": {
                    "code": "tool_execution_failed",
                    "message": (
                        f"The {tool_name} search failed. "
                        "Do not present search results because it fails."
                    ),
                }
            }

    @staticmethod
    def _extract_sources(
        tool_name: str,
        result: dict,
    ) -> list[SourceReference]:
        results = result.get("results", [])
        if not isinstance(results, list):
            return []

        source_type = "unit" if tool_name == "units_search" else "document"
        sources = []

        for item in results:
            if not isinstance(item, dict):
                continue

            payload = item.get("payload", item)

            if source_type == "unit":
                source_id = payload.get("unit_id")
            else:
                source_id = payload.get("filename")

            if source_id is None:
                continue

            score = item.get("rerank_score", item.get("score"))
            if score is None:
                score = item.get("qdrant_score")

            sources.append(
                SourceReference(
                    source_type=source_type,
                    source_id=str(source_id),
                    score=float(score) if score is not None else None,
                )
            )

        return sources

    @staticmethod
    def _unique_sources(
        sources: list[SourceReference],
    ) -> list[SourceReference]:
        unique: dict[tuple[str, str], SourceReference] = {}

        for source in sources:
            key = (source.source_type, source.source_id)
            existing = unique.get(key)

            if existing is None or (
                source.score is not None
                and (existing.score is None or source.score > existing.score)
            ):
                unique[key] = source

        return list(unique.values())


if __name__ == "__main__":
    from tools import UnitsSearchArguments, DocumentsSearchArguments, units_search, documents_search
    from config import AgentConfig

    # 1) config of Agent parameters (for vLLM request) + LLMClient
    config = AgentConfig()
    client = LLMClient(cfg=config)

    # 2) Tool Registery: to add tools
    tool_registry = ToolRegistry(
        units_limit=10,
        doc_limit=5,
        reranker_limit=3,
    )
    tool_registry.register(
        name="units_search",
        description=(
            "Search property listings, availability, and prices. Use this for "
            "requests about finding or comparing units. Pass the full search "
            "intent in query and any explicit city, district, type, price, or bedroom."
        ),
        arguments_model=UnitsSearchArguments,
        handler=units_search,
    )
    tool_registry.register(
        name="documents_search",
        description=(
            "if payment plan or purchase policy is asked in the query "
            "Pass the document question in query "
            "and known document type or location filters if applicable."
        ),
        arguments_model=DocumentsSearchArguments,
        handler=documents_search,
    )

    # 3) Persistent conversational memory
    from sqlalchemy.orm import sessionmaker
    from services.gateway.models import get_engine
    db_session_factory = sessionmaker(
            bind=get_engine(),
            expire_on_commit=False,
        )
    memory = DatabaseConversationMemory(session_factory=db_session_factory, max_turns=8)

    agent = AgentOrchestrator(
        llm_client=client,
        registry=tool_registry,
        memory=memory,
    )

    session_id = input("Session ID (leave blank for a new conversation): ").strip()
    question = input("Ask Atlas: ").strip()
    answer = agent.run(session_id=session_id or None, user_query=question)
    print(answer.model_dump_json(indent=2))