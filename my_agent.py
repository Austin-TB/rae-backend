import os
from typing import Dict, Any
import cmath
import numpy as np
from dotenv import load_dotenv 
from langchain_groq import ChatGroq
from langgraph.graph import START, StateGraph, MessagesState
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader, YoutubeLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
import pytesseract
from python_interpreter import run_python_script
from PIL import Image

load_dotenv(override=True)

@tool
def calculator(q:str) -> str:
    """
    A simple calculator tool that can perform basic arithmetic operations.
    Args:
        q (str): A string containing a mathematical expression in valid python syntax to evaluate.
    Returns:
        A string describing the available operations.
    """
    return eval(q)

    """
    Get the square root of a number.

    Args:
        a (float): the number to get the square root of
    """
    if a >= 0:
        return a**0.5
    return cmath.sqrt(a)
##-----------------------------------------------------------------------------------------##

@tool
def reverse_string(string: str) -> str:
    """
    Reverse a string.
    Args:
        string (str): The string to reverse.
    """
    return string[::-1]

##-----------------------------------------------------------------------------------------##

@tool(description="A tool to search Wikipedia for a query and return maximum 2 results.")
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 2 results.

    Args:
        query: The search query."""
    wiki_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_wiki_docs = "\n\n---\n\n".join(
        [
            f'{doc.metadata["title"]}\n{doc.page_content}\n-----------\n'
            for doc in wiki_docs
        ]
    )
    return formatted_wiki_docs

@tool(description="A tool to search the web for a query and return maximum 3 results.")
def web_search(query: str) -> str:
    """Search Tavily for a query and return maximum 3 results.

    Args:
        query: The search query."""
    tool = TavilySearchResults(max_results=3)
    search_docs = tool.invoke({'query': query})
    formatted_search_docs = "\n\n-----------\n\n".join(
        [
            f'{doc["title"]}\n{doc["url"]}\n{doc["content"]}\n-----------\n'
            for doc in search_docs
        ]
    )
    return formatted_search_docs

@tool(description="A tool to search Arxiv for a topic and return 2 paper summaries.")
def arxiv_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 result.
    Use this tool if the question is about a scientific paper.

    Args:
        query: The search query."""
    arxiv_docs = ArxivLoader(query=query, load_max_docs=2).load()
    # formatted_wiki_docs_full = "\n\n---\n\n".join(
    #     [
    #         f'{doc.metadata["Title"]}\n{doc.page_content}\n-----------\n'
    #         for doc in arxiv_docs
    #     ])
    formatted_wiki_docs_summaries = "\n\n---\n\n".join(
    [
        f'{doc.metadata["Title"]}\n{doc.metadata["Summary"]}\n-----------\n'
        for doc in arxiv_docs
    ]
    )
    return formatted_wiki_docs_summaries

@tool(description="A tool to extract text from an image using OCR library pytesseract.")
def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using OCR library pytesseract (if available).
    Args:
        image_path (str): the path to the image file.
    """
    try:
        # Open the image
        image = Image.open(image_path)

        # Extract text from the image
        text = pytesseract.image_to_string(image)

        return f"Extracted text from image:\n\n{text}"
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"

@tool(description="A tool to analyze a CSV file using pandas and answer a question about it.")
def analyze_csv_file(file_path: str, query: str) -> str:
    """
    Analyze a CSV file using pandas and answer a question about it.
    Args:
        file_path (str): the path to the CSV file.
        query (str): Question about the data
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Run various analyses based on the query
        result = f"CSV file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"

        # Add summary statistics
        result += "Summary statistics:\n"
        result += str(df.describe())

        return result

    except Exception as e:
        return f"Error analyzing CSV file: {str(e)}"

@tool(description="A tool to analyze an Excel file using pandas and answer a question about it.")
def analyze_excel_file(file_path: str, query: str) -> str:
    """
    Analyze an Excel file using pandas and answer a question about it.
    Args:
        file_path (str): the path to the Excel file.
        query (str): Question about the data
    """
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)

        # Run various analyses based on the query
        result = (
            f"Excel file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        )
        result += f"Columns: {', '.join(df.columns)}\n\n"

        # Add summary statistics
        result += "Summary statistics:\n"
        result += str(df.describe())

        return result

    except Exception as e:
        return f"Error analyzing Excel file: {str(e)}"

@tool
def execute_python_script(file_path: str) -> str:
    """
    Execute a Python script and return the output.
    Args:
        file_path (str): The local path to the Python script.
    Returns:
        A string containing the output of the script.
    """
    return run_python_script(file_path)

##-----------------------------------------------------------------------------------------##

@tool(description="A tool to scrape a website and return the text.")
def scrape_website(url: str) -> str:
    """
    Scrape a website and return the text.
    Args:
        url (str): The URL of the website to scrape.
    Returns:
        A string containing the text of the website.
    """
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs[0].page_content

@tool(description="A tool to scrape a youtube video and return the text.")
def scrape_youtube(url: str) -> str:
    """
    Scrape a youtube video and return the text.
    Args:
        url (str): The URL of the youtube video to scrape.
    Returns:
        A string containing the text of the youtube video.
    """
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
    docs = loader.load()
    return docs[0].page_content

@tool
def retriever_tool(query: str) -> Dict[str, Any]:
    """
    A tool to retrieve relevant documents from a vector store based on a query.
    Args:
        query (str): The query to search for.
    Returns:
        A dictionary containing the retrieved documents.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)
    
    docs = vector_store.similarity_search(query, k=3)

    formatted_local_docs = "\n\n-----------\n\n".join(
    [
        doc.page_content + "\n-----------\n"
        for doc in docs
    ])
    return formatted_local_docs

tools =[
    calculator,
    web_search,
    wiki_search,
    arxiv_search,
    extract_text_from_image,
    analyze_csv_file,
    analyze_excel_file,
    execute_python_script,
    reverse_string,
    scrape_website,
    scrape_youtube,
    retriever_tool
]

def build_agent(provider: str = "google"):
    if provider == "qwen":
        llm = ChatGroq(model="qwen-qwq-32b", temperature=0)
    elif provider == "llama":
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    elif provider == "google":
        llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,  
            )

    llm_with_tools = llm.bind_tools(tools)
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    system_prompt_path = os.path.join(script_dir, "system_prompt.txt")
    
    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()
    sys_msg = SystemMessage(content = system_prompt)

    def assistant(state: MessagesState):
        
        # Prepare messages for the LLM: System Prompt + current history
        messages_for_llm_invocation = [sys_msg] + state["messages"]
        
        # Invoke LLM with the system prompt and current history
        ai_response_message = llm_with_tools.invoke(messages_for_llm_invocation)
        
        # Return only the new AI message to be appended to the state
        return {"messages": [ai_response_message]}


    graph = StateGraph(MessagesState)    
    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "assistant")
    graph.add_conditional_edges(
        "assistant",
        tools_condition
    )
    graph.add_edge("tools", "assistant")

    agent = graph.compile()
    return agent

# if __name__ == "__main__":
#     question = "What is the latest news on Joe Biden?"
#     messages = [HumanMessage(content=question)]
#     agent = build_agent()
#     messages = agent.invoke({"messages": messages})
#     for m in messages["messages"]:
#         m.pretty_print()