import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import time

app = FastAPI()

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    
    last_msg = messages[-1].get("content", "") if messages else ""
    
    response_text = ""
    
    if "Is hallucinated (Yes/No):" in last_msg:
        response_text = "No"
    elif "Is unsafe (Yes/No):" in last_msg:
        response_text = "No"
    else:
        response_text = "The telephone was invented by Alexander Graham Bell in 1876. (Note: This is a response from the Local Mock API running on port 8080 to demonstrate zero rate-limits!)"

    return JSONResponse(content={
        "id": "chatcmpl-mock123",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": data.get("model", "mock-model"),
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text,
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "total_tokens": 20
        }
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
