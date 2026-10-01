from pathlib import Path
import sys
from typing import Callable

from sqlalchemy.orm import Session, sessionmaker

if __package__:
    from .schemas import ChatMessage
    from ..gateway.models import Conversation, get_engine
else:
    from schemas import ChatMessage

    _REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPOSITORY_ROOT))

    from services.gateway.models import Conversation, get_engine


class DatabaseConversationMemory:
    """Persist a bounded window of user/assistant turns in PostgreSQL."""

    def __init__(
        self,
        session_factory: Callable[[], Session] | None = None,
        max_turns: int = 8,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")

        self._max_messages = max_turns * 2
        self._session_factory = session_factory or sessionmaker(
            bind=get_engine(),
            expire_on_commit=False,
        )

    def get_history(self, session_id: str) -> list[ChatMessage]:
        self._validate_session_id(session_id)

        with self._session_factory() as session:
            conversations = (
                session.query(Conversation)
                .filter(
                    Conversation.session_id == session_id,
                    Conversation.role.in_(("user", "assistant")),
                )
                .order_by(Conversation.created_at.desc(), Conversation.id.desc())
                .limit(self._max_messages)
                .all()
            )

        return [
            ChatMessage(role=message.role, content=message.content)
            for message in reversed(conversations)
        ]

    def remember_turn(
        self,
        session_id: str,
        messages: list[ChatMessage],
    ) -> None:
        self._validate_session_id(session_id)
        if len(messages) != 2 or [message.role for message in messages] != [
            "user",
            "assistant",
        ]:
            raise ValueError(
                "A turn must contain one user message and one assistant message"
            )
        if any(message.content is None or message.tool_calls for message in messages):
            raise ValueError(
                "Only completed user/assistant text messages can be persisted"
            )

        with self._session_factory() as session:
            session.add_all(
                Conversation(
                    session_id=session_id,
                    role=message.role,
                    content=message.content,
                )
                for message in messages
            )
            session.flush()

            stale_message_ids = (
                session.query(Conversation.id)
                .filter(
                    Conversation.session_id == session_id,
                    Conversation.role.in_(("user", "assistant")),
                )
                .order_by(Conversation.created_at.desc(), Conversation.id.desc())
                .offset(self._max_messages)
                .all()
            )
            if stale_message_ids:
                session.query(Conversation).filter(
                    Conversation.id.in_(
                        [message_id for (message_id,) in stale_message_ids]
                    )
                ).delete(synchronize_session=False)

            session.commit()

    def clear_session(self, session_id: str) -> None:
        self._validate_session_id(session_id)

        with self._session_factory() as session:
            (
                session.query(Conversation)
                .filter(Conversation.session_id == session_id)
                .delete(synchronize_session=False)
            )
            session.commit()

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not session_id or not session_id.strip():
            raise ValueError("session_id must not be empty")
        if len(session_id) > 100:
            raise ValueError("session_id must not exceed 100 characters")
