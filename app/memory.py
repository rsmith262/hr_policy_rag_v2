from typing import Callable
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory

def make_history_getter(redis_url: str | None, ttl_seconds: int) -> Callable[[str], ChatMessageHistory]:
    """Return a function that gives a ChatMessageHistory for a session_id.
    Uses Redis if redis_url is valid; else falls back to in-process memory (dev only)."""
    # Normalize and check that redis_url is valid
    if redis_url:
        # Simple validation: must start with redis:// or rediss://
        if redis_url.startswith("redis://") or redis_url.startswith("rediss://"):
            def _get(session_id: str):
                return RedisChatMessageHistory(
                    session_id=f"chat:{session_id}",
                    url=redis_url,
                    ttl=ttl_seconds
                )
            return _get

    # fallback in-memory
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
