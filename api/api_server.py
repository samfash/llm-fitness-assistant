from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from core.rag_with_semantic_retriever import retrieve_context
from core.llm_engine import generate_response
from core.memory import memory_store, get_user_context
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="LLM Fitness Assistant API",
    description="Backend API for the LLM Fitness Assistant using LangChain + LoRA + Chroma.",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    user_id: str
    prompt: str

@app.post("/chat")
async def chat(query: Query):
    user_context = get_user_context(query.user_id)
    user_input = query.prompt.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Empty query received.")

    retrieved_knowledge = retrieve_context(user_input)

    final_prompt = f"""
    You are a personal fitness assistant.
    Use user’s past logs and external data to respond helpfully.

    Past context: {user_context}
    Retrieved data: {retrieved_knowledge}
    User input: {user_input}
    """

    response = generate_response(final_prompt)
    memory_store(query.user_id, user_input, response)

    try:
        return {"assistant_reply": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/")
def root():
    return {"message": "LLM Fitness Assistant API is running 🚀"}
