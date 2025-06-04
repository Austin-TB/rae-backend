"""FastAPI Backend for Rae Chatbot"""
import os
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, parse_obj_as
from typing import List, Dict, Any
import json
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from my_agent import build_agent
import uvicorn
import shutil

app = FastAPI(title="Rae Chat API", version="1.0.0")

# Add CORS middleware
allowed_origins=[
    "http://localhost:5173",
    "http://localhost:3001",
    "https://rae-frontend.vercel.app",
    "https://chatwithrae.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
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

@app.options("/chat")
async def chat_options():
    """Handle OPTIONS requests for CORS preflight"""
    return {"message": "OK"}

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")

@app.post("/chat", response_model=ChatResponse)
async def chat(
    message: str = Form(...),
    conversation_history_str: str = Form("[]"),
    file: UploadFile = File(None)
):
    try:
        user_message_content = message
        file_path_message = ""

        if file:
            filename = os.path.basename(file.filename)
            file_path = os.path.join(UPLOAD_DIR, filename)
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            print(f"File saved to: {file_path}")
            file_path_message = f"\\nfile_path:{file_path}"

        full_user_message = f"{user_message_content}{file_path_message}"

        try:
            parsed_history_data = json.loads(conversation_history_str)
            conversation_history: List[ChatMessage] = parse_obj_as(List[ChatMessage], parsed_history_data)
        except json.JSONDecodeError:
            print("Error decoding conversation_history_str JSON.")
            conversation_history = []
        except Exception as e:
            print(f"Error parsing conversation_history: {e}")
            conversation_history = []

        langchain_history = convert_to_langchain_messages(conversation_history)
        langchain_history.append(HumanMessage(content=full_user_message))
        response_content = agent(langchain_history)

        return ChatResponse(response=response_content)
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("\n" + "-"*30 + "Rae Backend" + "-"*30)

    # Create or clear the upload directory
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    print(f"Upload directory '{UPLOAD_DIR}' ensured to be clean and exists.")

    server_host = os.getenv("BACKEND_HOST", "0.0.0.0")
    server_port = int(os.getenv("BACKEND_PORT", "8000"))
    uvicorn.run(app, host=server_host, port=server_port, log_level="info") 