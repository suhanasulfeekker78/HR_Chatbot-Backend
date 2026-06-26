import json
from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse
from agents.agent import hr_agent_executor

router = APIRouter(tags=["Chat Engine"])

@router.post("/chat/stream")
async def handle_chat_stream(request: Request):
    auth_header = request.headers.get("Authorization", "")
    body = await request.json()
    user_prompt = body.get("content", "")
    
    async def sse_generator():
        try:
            async for chunk in hr_agent_executor.astream(
                {"input": user_prompt, "chat_history": []},
                config={"configurable": {"auth_token": auth_header}}
            ):
                if "messages" in chunk:
                    for msg in chunk["messages"]:
                        if hasattr(msg, "content") and msg.content:
                            yield {"data": json.dumps({"content": msg.content})}
                            
            yield {"data": "[DONE]"}
            
        except Exception as e:
            yield {"data": json.dumps({"content": f"\n\n*System Connection Interrupt: {str(e)}*"})}
            yield {"data": "[DONE]"}

    return EventSourceResponse(sse_generator())