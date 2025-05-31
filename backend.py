"""FastAPI Backend for Rae Chatbot"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from my_agent import build_agent
import uvicorn

app = FastAPI(title="Rae Chat API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: str = None

class ChatRequest(BaseModel):
    message: str
    conversation_history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    response: str
    conversation_id: str = None

class BasicAgent:
    """A langgraph agent."""
    def __init__(self):
        print("BasicAgent initialized.")
        self.graph = build_agent()

    def __call__(self, conversation_messages: list) -> str:
        print(f"Agent received {len(conversation_messages)} messages.")
        
        response_data = self.graph.invoke({"messages": conversation_messages})
        
        for m in response_data["messages"]:
            m.pretty_print()
            
        answer = response_data['messages'][-1].content
        return answer

# Global agent instance
agent = BasicAgent()

def convert_to_langchain_messages(history: List[ChatMessage]) -> List[BaseMessage]:
    """Convert chat history to LangChain message format"""
    messages = []
    for msg in history:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))
    return messages

@app.get("/")
async def root():
    return {"message": "Rae Chat API is running!"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        langchain_history = convert_to_langchain_messages(request.conversation_history)
        langchain_history.append(HumanMessage(content=request.message))
        response_content = agent(langchain_history)

        return ChatResponse(response=response_content)
        # return ChatResponse(response="Henlo")
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("\n" + "-"*30 + "Rae Backend" + "-"*30)
    server_host = os.getenv("BACKEND_HOST", "0.0.0.0")
    server_port = int(os.getenv("BACKEND_PORT", "8000"))
    uvicorn.run(app, host=server_host, port=server_port, log_level="info") 