from collections import deque
from threading import RLock

from schemas import ChatMessage


class ConversationMemory:
    """A thread-safe, in-process window of completed conversation turns."""

    def __init__(self, max_turns: int = 8) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")

        self._max_turns = max_turns
        self._sessions: dict[str, deque[tuple[ChatMessage, ...]]] = {}
        self._lock = RLock()

    def get_history(self, session_id: str) -> list[ChatMessage]:
        """Return a copy of recent turns, excluding the system prompt."""
        self._validate_session_id(session_id)

        with self._lock:
            turns = self._sessions.get(session_id, ())
            return [
                message.model_copy(deep=True)
                for turn in turns
                for message in turn
            ]

    def remember_turn(
        self,
        session_id: str,
        messages: list[ChatMessage],
    ) -> None:
        """Store one completed turn: user message, optional tool exchange, final answer."""
        self._validate_session_id(session_id)

        if not messages:
            raise ValueError("A turn must contain at least one message")

        turn = tuple(
            message.model_copy(deep=True)
            for message in messages
        )

        if turn[0].role != "user":
            raise ValueError("A turn must start with a user message")

        if any(message.role == "system" for message in turn):
            raise ValueError("System messages belong in the prompt, not memory")

        final_message = turn[-1]
        if final_message.role != "assistant" or final_message.content is None:
            raise ValueError("A completed turn must end with an assistant answer")

        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = deque(maxlen=self._max_turns)

            self._sessions[session_id].append(turn)

    def clear_session(self, session_id: str) -> None:
        """Forget all stored turns for one session."""
        self._validate_session_id(session_id)

        with self._lock:
            self._sessions.pop(session_id, None)

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not session_id or not session_id.strip():
            raise ValueError("session_id must not be empty")

"""How to use memory and add turns?
memory = ConversationMemory(max_turns=8)

history = memory.get_history(session_id)

# Build the vLLM messages:
# system prompt + history + current user message
messages = [
    ChatMessage(role="system", content=SYSTEM_PROMPT),
    *history,
    ChatMessage(role="user", content=user_query),
]

# After orchestration finishes, store the complete current turn:
memory.remember_turn(
    session_id,
    [
        ChatMessage(role="user", content=user_query),
        *assistant_and_tool_messages_from_this_turn,
        ChatMessage(role="assistant", content=final_answer),
    ],
)
"""

