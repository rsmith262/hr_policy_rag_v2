from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.models import ChatRequest, ChatResponse, Citation
from app.memory import make_history_getter, wrap_with_history
from app.rag import answer_with_citations, prompt, llm

def require_api_key(x_api_key: str | None = Header(default=None)):
    want = settings.api_key
    if want and x_api_key != want:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

app = FastAPI(title="RAG Backend (Azure AI Search)", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

get_history = make_history_getter(settings.redis_url or None, settings.redis_ttl)
base_chain = (prompt | llm)
mem_chain = wrap_with_history(base_chain, get_history)

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, _: bool = Depends(require_api_key)):
    session_id = req.session_id or "anonymous"
    inputs = {"input": req.input}

    ai_msg, cites = answer_with_citations(inputs, session_id=session_id)


    return ChatResponse(
        reply=getattr(ai_msg, "content", str(ai_msg)),
        citations=[Citation(**c) for c in cites]
    )
