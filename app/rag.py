from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_core.documents import Document
from app.config import settings
from app.memory import make_history_getter, wrap_with_history

import logging
logger = logging.getLogger("app")

# LLM
llm = AzureChatOpenAI(
    azure_endpoint=settings.aoai_endpoint,
    api_key=settings.aoai_key,
    azure_deployment=settings.aoai_deployment,
    api_version=settings.aoai_api_version,
    temperature=0
)

# Embeddings
emb = AzureOpenAIEmbeddings(
    azure_endpoint=settings.aoai_endpoint,
    api_key=settings.aoai_key,
    azure_deployment=settings.embed_deployment,
    api_version=settings.embed_api_version,
)

# Vector store / retriever
vectorstore = AzureSearch(
    azure_search_endpoint=settings.search_endpoint,
    azure_search_key=settings.search_key,
    index_name=settings.search_index,
    embedding_function=emb.embed_query,  # used for query embeddings
)


retriever = vectorstore.as_retriever(search_type="hybrid", k=4)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use CONTEXT and chat HISTORY. If unsure, say you don't know."),
    MessagesPlaceholder("history"),
    ("system", "CONTEXT:\n{context}"),
    ("human", "{input}")
])

def _join(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)

def _cites(docs: List[Document]) -> list[dict]:
    out = []
    for d in docs:
        out.append({
            "source": str(d.metadata.get("source", "unknown")),
            "page": d.metadata.get("page") if isinstance(d.metadata.get("page"), int) else None,
            "url": d.metadata.get("url"),
        })
    return out

def fetch_context(inputs: Dict[str, Any]) -> Dict[str, Any]:
    docs = retriever.invoke(inputs["input"])
    return {"context": _join(docs), "docs": docs}

# Build RAG chain
rag_chain = (RunnablePassthrough.assign(context_bundle=fetch_context)
             .assign(context=lambda x: x["context_bundle"]["context"])
             | prompt
             | llm)

# bring in history management
# Pass settings.redis_url directly (even if empty), memory.py handles fallback
get_history = make_history_getter(settings.redis_url, settings.redis_ttl)
mem_chain = wrap_with_history(prompt | llm, get_history)

def answer_with_citations(inputs: Dict[str, Any], session_id: str = "anonymous"):
    """
    1) Retrieve context from Azure AI Search
    2) Generate with prompt+history using that context
    3) Return citations from the retrieved docs
    """
    user_input = inputs.get("input", "")

    # Step 1: retrieve
    ctx_bundle = fetch_context({"input": user_input})   # -> {"context": str, "docs": List[Document]}
    ctx_text = ctx_bundle.get("context", "")
    docs = ctx_bundle.get("docs", [])

    # (optional logging)
    try:
        logger.info("Retrieval returned %d docs", len(docs))
    except Exception:
        pass

    # Step 2: generate with history + injected context
    ai_msg = mem_chain.invoke(
        {"input": user_input, "context": ctx_text},
        config={"configurable": {"session_id": session_id}},
    )

    # Step 3: citations from retrieved docs
    return ai_msg, _cites(docs)
