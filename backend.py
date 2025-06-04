"""FastAPI Backend for Rae Chatbot"""
import os
from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from my_agent import build_agent
import uvicorn
import json
import shutil # Added for directory operations
from pathlib import Path # Added for path manipulation

app = FastAPI(title="Rae Chat API", version="1.0.0")

# Define a temporary directory for uploads
TEMP_UPLOAD_DIR = Path("rae_temp_uploads")

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

class ChatResponse(BaseModel):
    response: str
    conversation_id: str = None

class BasicAgent:
    """A langgraph agent."""
    def __init__(self):
        print("BasicAgent initialized.")
        self.graph = build_agent()

    def __call__(self, conversation_messages: list) -> str:
        print(f"Agent received {len(conversation_messages)} messages:")
        for i, msg in enumerate(conversation_messages):
            print(f"  Message {i}: role={msg.type if hasattr(msg, 'type') else 'unknown'}, content='{msg.content}'")
        
        try:
            response_data = self.graph.invoke({"messages": conversation_messages})
            
            print("Agent response data:", response_data)
            if not response_data or "messages" not in response_data or not response_data["messages"]:
                print("Error: Agent returned empty or invalid response structure.")
                return "Sorry, I received an empty or invalid response from the agent."

            for m in response_data["messages"]:
                m.pretty_print()
                
            answer = response_data['messages'][-1].content
            return answer
        except Exception as e:
            print(f"Error during agent invocation or processing: {str(e)}")
            # Optionally, re-raise or return a specific error message
            # For now, let's return a clear error message to the user
            return f"Sorry, an error occurred while processing your request with the agent: {str(e)}"

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

@app.post("/chat", response_model=ChatResponse)
async def chat(
    message: str = Form(...),
    conversation_history_str: str = Form(..., alias="conversation_history"),
    file: Optional[UploadFile] = File(None)
):
    try:
        try:
            history_data = json.loads(conversation_history_str)
            actual_conversation_history = [ChatMessage(**item) for item in history_data]
        except json.JSONDecodeError:
            print(f"Error decoding conversation_history_str: {conversation_history_str}")
            raise HTTPException(status_code=400, detail="Invalid conversation_history format.")
        except Exception as e: 
            print(f"Error processing conversation_history: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error in conversation_history: {str(e)}")

        langchain_history = convert_to_langchain_messages(actual_conversation_history)
        
        user_message_content = message
        file_path_for_agent = None

        if file:
            # Ensure the temporary directory exists
            TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

            # If it's the first message (empty history), clear the temp directory
            if not actual_conversation_history:
                print(f"First message in conversation, clearing {TEMP_UPLOAD_DIR}")
                for item in TEMP_UPLOAD_DIR.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
            
            # Sanitize filename to prevent security issues, though for this controlled temp dir, less critical
            # For production, consider more robust sanitization or using UUIDs for filenames
            # filename = secure_filename(file.filename) # Example if using a utility like Werkzeug's
            filename = Path(file.filename).name # Basic sanitization to get only the filename part
            save_path = TEMP_UPLOAD_DIR / filename

            try:
                contents = await file.read()
                with open(save_path, "wb") as buffer:
                    buffer.write(contents)
                await file.seek(0) # Reset file pointer, good practice
                file_path_for_agent = str(save_path.resolve()) # Get absolute path
                print(f"File {file.filename} saved to {file_path_for_agent}")
                user_message_content += f" [File Path: {file_path_for_agent}]"
            except Exception as e:
                print(f"Error saving file {file.filename}: {str(e)}")
                # Decide if you want to raise an error or just proceed without the file
                user_message_content += f" [Error saving file: {file.filename}]"
                # Not raising HTTPException here to allow chat to continue if file saving fails

        langchain_history.append(HumanMessage(content=user_message_content))
        response_content = agent(langchain_history)

        return ChatResponse(response=response_content)
        # return ChatResponse(response="Henlo")
        
    except HTTPException: 
        raise
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("\n" + "-"*30 + "Rae Backend" + "-"*30)
    # Ensure the temp directory is created on startup if it doesn't exist
    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Temporary upload directory: {TEMP_UPLOAD_DIR.resolve()}")

    server_host = os.getenv("BACKEND_HOST", "0.0.0.0")
    server_port = int(os.getenv("BACKEND_PORT", "8000"))
    uvicorn.run(app, host=server_host, port=server_port, log_level="info") 