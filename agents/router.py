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

    session_key = auth_header.replace("Bearer ", "").strip()
    if not session_key:
        session_key = "anonymous_fallback_session"

    async def sse_generator():
        try:
            async for chunk_msg, metadata in hr_agent_executor.astream(
                {"messages": [("user", user_prompt)]},
                config={
                    "configurable": {
                        "auth_token": auth_header, 
                        "thread_id": session_key  
                    }
                },
                stream_mode="messages"
            ):
                if chunk_msg.content and metadata.get("langgraph_node") == "agent":
                    yield {"data": json.dumps({"content": chunk_msg.content})}
                            
            yield {"data": "[DONE]"}
            
        except Exception as e:
            yield {"data": json.dumps({"content": f"\n\n*System Connection Interrupt: {str(e)}*"})}
            yield {"data": "[DONE]"}

    return EventSourceResponse(sse_generator())