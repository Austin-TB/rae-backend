"""Simple Question Fetcher and Display App"""
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from my_agent import build_agent
import os

langchain_formatted_history = []

class BasicAgent:
    """A langgraph agent."""
    def __init__(self):
        print("BasicAgent initialized.")
        self.graph = build_agent()

    def __call__(self, conversation_messages: list) -> str:
        print(f"Agent received {len(conversation_messages)} messages.")
        if conversation_messages:
            pass

        response_data = self.graph.invoke({"messages": conversation_messages})
        
        for m in response_data["messages"]:
            m.pretty_print()
            
        answer = response_data['messages'][-1].content
        return answer

def agent_response(current_user_message: str, _):
    agent = BasicAgent()
    langchain_formatted_history.append(HumanMessage(content=current_user_message))
    response_content = agent(langchain_formatted_history)
    langchain_formatted_history.append(AIMessage(content=response_content))
    yield response_content
    # yield "Hello"

with gr.Blocks(css_paths="./style.css") as demo:
    gr.ChatInterface(
        agent_response,
        chatbot=gr.Chatbot(height=600, type='messages', elem_id="chatbot-container"),
        textbox=gr.Textbox(placeholder="Ask a question...", container=False, scale=7, elem_id="textbox-container"),
        title="Rae",
        description="Making it work",
        examples=[["Explain this youtube video: https://www.youtube.com/watch?v=Qw6b1a2d3e4"],["what is the Capital of France?"]],
        cache_examples=False,
        type="messages",
        # additional_inputs=[gr.UploadButton(label="Upload File", file_types=["any"])],
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + "Rae" + "-"*30)
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    enable_debug = os.getenv("GRADIO_DEBUG", "False").lower() == "true"
    demo.launch(server_name=server_name, server_port=server_port, debug=enable_debug, share=False)