import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from agents.router import router as chat_router
from hr_policy.router import router as hr_policy_router

app = FastAPI(title="HR Chatbot Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(hr_policy_router)

if __name__ == "__main__":
    uvicorn.run("main.py:app", host=settings.HOST, port=settings.PORT, reload=True)