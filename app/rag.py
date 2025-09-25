from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_core.documents import Document
from app.config import settings

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

# added debugging
import logging
logging.warning(f"Search endpoint: {settings.search_endpoint}")
logging.warning(f"Search index: {settings.search_index}")
logging.warning(f"Search key starts with: {settings.search_key[:5]}")
# End deubugging

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
    docs = retriever.get_relevant_documents(inputs["input"])
    return {"context": _join(docs), "docs": docs}

# Build RAG chain
rag_chain = (RunnablePassthrough.assign(context_bundle=fetch_context)
             .assign(context=lambda x: x["context_bundle"]["context"])
             | prompt
             | llm)

def answer_with_citations(inputs: Dict[str, Any]):
    ai_msg = rag_chain.invoke(inputs)
    docs = inputs["context_bundle"]["docs"]
    return ai_msg, _cites(docs)
