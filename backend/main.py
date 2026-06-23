from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(
    title="WikiMind RAG API",
    description="Backend API for the Tri-Modal Hybrid Agentic RAG Pipeline",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    """Health check endpoint to verify backend service status."""
    return JSONResponse(content={"status": "healthy", "service": "WikiMind Backend"})

@app.post("/chat")
async def chat_endpoint():
    """Placeholder for the primary RAG chat endpoint."""
    return JSONResponse(content={"message": "Chat endpoint initialized."})
