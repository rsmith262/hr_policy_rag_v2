from typing import Callable
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory

def make_history_getter(redis_url: str | None, ttl_seconds: int) -> Callable[[str], ChatMessageHistory]:
    """Return a function that gives a ChatMessageHistory for a session_id.
    Uses Redis if redis_url is provided; else falls back to in-process memory (dev only)."""
    if redis_url:
        def _get(session_id: str):
            return RedisChatMessageHistory(
                session_id=f"chat:{session_id}",
                url=redis_url,
                ttl=ttl_seconds
            )
        return _get

    _mem = {}
    def _get(session_id: str):
        if session_id not in _mem:
            _mem[session_id] = ChatMessageHistory()
        return _mem[session_id]
    return _get

def wrap_with_history(chain, get_history):
    """Wrap any Runnable (prompt|llm chain) with persistent chat history management."""
    return RunnableWithMessageHistory(
        chain,
        get_history,
        input_messages_key="input",
        history_messages_key="history",
    )

# Simple in-process memory (alternative quick option)
def simple_buffer_memory():
    return ConversationBufferMemory(return_messages=True)
