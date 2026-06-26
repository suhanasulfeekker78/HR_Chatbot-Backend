import json
import traceback
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
                node_name = metadata.get("langgraph_node")
                if hasattr(chunk_msg, "tool_calls") and chunk_msg.tool_calls:
                    for tool_call in chunk_msg.tool_calls:
                        print(f"\n[AGENT ACTION] Calling Tool: '{tool_call['name']}' with args: {tool_call['args']}")

                if node_name == "tools" and chunk_msg.content:
                    print(f"[TOOL RESPONSE] Result from tool:\n{chunk_msg.content}\n")
                if node_name == "model" and chunk_msg.content:
                    print(chunk_msg.content, end="", flush=True)
                    yield {"data": json.dumps({"content": chunk_msg.content})}
                            
            yield {"data": "[DONE]"}
            
        except Exception as e:
            print("\n" + "="*50)
            print("AGENT EXECUTION ERROR DETECTED IN BACKEND:")
            print(f"Error Type: {type(e).__name__}")
            print(f"Details: {str(e)}")
            print("="*50)
            traceback.print_exc()
            print("="*50 + "\n")
            yield {"data": json.dumps({"content": f"\n\nSystem Connection Interrupt"})}
            yield {"data": "[DONE]"}

    return EventSourceResponse(sse_generator())